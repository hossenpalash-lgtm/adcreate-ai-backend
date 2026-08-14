from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from google import genai
from google.genai import types as genai_types
from supabase import create_client
from postgrest.exceptions import APIError
from dotenv import load_dotenv
from typing import Optional, Callable, TypeVar
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import os
import json
import re
import time
import logging
import base64
import requests
import sentry_sdk
from datetime import datetime, timedelta, timezone

# -----------------------
# LOGGING
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("adcreate")

# -----------------------
# LOAD ENV
# -----------------------
load_dotenv()

# -----------------------
# ERROR TRACKING
# -----------------------
# Optional — only enabled when SENTRY_DSN is set, so local dev without a
# DSN configured doesn't try to report anywhere.
SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
if SENTRY_DSN:
    sentry_sdk.init(dsn=SENTRY_DSN, send_default_pii=False)

# -----------------------
# RETRY HELPER
# -----------------------
# Only used for read-only / idempotent calls (OpenAI/Gemini calls, Supabase
# selects). Writes (insert/update) are deliberately NOT retried here —
# blindly retrying an insert on a timeout risks double-writing, which is
# worse than surfacing a clean error.
T = TypeVar("T")

RETRYABLE_OPENAI_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)


def with_retry(fn: Callable[[], T], attempts: int = 3, delay: float = 1.0, exceptions=(Exception,)) -> T:
    last_exc: Exception = Exception("with_retry called with attempts <= 0")
    for i in range(attempts):
        try:
            return fn()
        except exceptions as e:
            last_exc = e
            logger.warning("Attempt %d/%d failed: %s", i + 1, attempts, e)
            if i < attempts - 1:
                time.sleep(delay * (i + 1))
    raise last_exc

# -----------------------
# CREATE APP
# -----------------------
app = FastAPI(
    title="AdCreate.AI Backend",
    description="API for generating AI ad copy, banner images, and weekly content plans for small businesses.",
    version="0.1.0"
)


# Render (and most PaaS hosts) sit the app behind a reverse proxy, so
# request.client.host is the proxy's IP, not the real caller's — every
# request would collapse onto one IP and rate limiting would apply
# globally instead of per-caller. Trust the proxy's X-Forwarded-For
# to recover the real client IP.
@app.middleware("http")
async def trust_x_forwarded_for(request: Request, call_next):
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        real_ip = forwarded_for.split(",")[0].strip()
        existing_port = request.scope["client"][1] if request.scope.get("client") else 0
        request.scope["client"] = (real_ip, existing_port)
    return await call_next(request)


# -----------------------
# RATE LIMITING
# -----------------------
# Every route here is auth-gated already, so anonymous abuse is capped by
# the 401 rejection itself — this is mainly about capping OpenAI/Gemini
# spend if a single account (or a leaked token) gets hammered, plus a
# generic ceiling everywhere else against runaway clients/bugs.
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # Match the rest of the API's error shape ({"detail": ...}) instead of
    # slowapi's default {"error": ...} — the frontend only ever looks for
    # "detail" when surfacing error messages.
    response = JSONResponse({"detail": "Too many attempts, please try again in a bit."}, status_code=429)
    return limiter._inject_headers(response, request.state.view_rate_limit)


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)

# -----------------------
# CORS MIDDLEWARE
# -----------------------
# Origins are env-driven (comma-separated) rather than hardcoded, since this
# app's real frontend domain isn't known yet at scaffold time — set
# ALLOWED_ORIGINS on the deployed backend once the frontend host is live.
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()] + [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# MODELS
# -----------------------
class AdCreditsOut(BaseModel):
    credits: int


class AdCaptionVariant(BaseModel):
    facebook_caption: str
    whatsapp_message: str


class AdGenerateResponse(BaseModel):
    captions: list[AdCaptionVariant]
    banner_image_base64: str
    credits_remaining: int


class AdImageVariantResponse(BaseModel):
    banner_image_base64: str
    credits_remaining: int


BUSINESS_CATEGORIES = {
    "retail", "restaurant_cafe", "health_beauty", "professional_services",
    "home_services", "real_estate", "automotive", "education_coaching",
    "fitness_sports", "events_entertainment", "ecommerce",
    "technology_software", "other",
}


class BusinessProfileOut(BaseModel):
    category: str


class SetBusinessProfileRequest(BaseModel):
    category: str

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        if v not in BUSINESS_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(BUSINESS_CATEGORIES)}")
        return v


class ContentPlanPostOut(BaseModel):
    day: str
    theme: str
    idea_text: str
    source_items: list[str]
    media_type: str
    status: str
    caption: Optional[str] = None
    whatsapp_message: Optional[str] = None
    image_base64: Optional[str] = None


class ContentPlanOut(BaseModel):
    id: str
    period_start: str
    period_end: str
    status: str
    posts: list[ContentPlanPostOut]


class GenerateContentPlanRequest(BaseModel):
    input_text: str


class SelectContentPlanPostRequest(BaseModel):
    caption: str
    whatsapp_message: str
    image_base64: Optional[str] = None


# -----------------------
# ENV / CLIENTS
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Optional — only the AI banner-image feature needs this. Not a hard
# startup requirement, since the rest of the app must keep working even
# before/without this one being configured.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    raise Exception("Missing OPENAI_API_KEY")
if not SUPABASE_URL:
    raise Exception("Missing SUPABASE_URL")
if not SUPABASE_KEY:
    raise Exception("Missing SUPABASE_KEY")

# Normalize Supabase URL for Python client
SUPABASE_URL = SUPABASE_URL.rstrip("/")
if SUPABASE_URL.lower().endswith("/rest/v1"):
    SUPABASE_URL = SUPABASE_URL[: -len("/rest/v1")]

client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0, max_retries=0)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

supabase = create_client(SUPABASE_URL, SUPABASE_KEY.strip())


def ensure_supabase_response(response, operation):
    if response is None:
        raise Exception(f"Supabase {operation} returned no response")
    if not hasattr(response, "data"):
        raise Exception(f"Supabase {operation} response missing data: {response!r}")
    if response.data is None:
        raise Exception(f"Supabase {operation} failed: {response!r}")
    return response


# -----------------------
# AUTH
# -----------------------
# Every endpoint (except the health check) requires a valid Supabase
# session. We delegate verification to Supabase's own /auth/v1/user
# endpoint rather than checking the JWT signature ourselves — this avoids
# needing another secret and always reflects live token state (revocation,
# expiry) instead of a cached assumption. The returned user id is used as
# owner_id to scope every query — this is the actual access-control
# boundary, since the backend talks to Supabase with a service-role key
# that bypasses Row Level Security. RLS is enabled as defense-in-depth,
# not as the primary enforcement.
def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Could not verify your session, please sign in again")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Could not verify your session, please sign in again")

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {token}"},
            timeout=10,
        )
    except requests.RequestException as e:
        logger.error("Auth check failed: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail="Could not verify your session, please try again")

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Your session has expired, please sign in again")

    user_id = resp.json().get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    return user_id


@app.get("/")
def home():
    return {"status": "AdCreate.AI backend is running"}


# -----------------------
# AD CREDITS
# -----------------------
# Every new user starts with a small free trial so they can experience the
# product before needing to pay — granted lazily the first time their
# ad_credits row is ever touched.
TRIAL_AD_CREDITS = 3


def _ensure_ad_credits_row(user_id: str) -> None:
    """Creates this user's ad_credits row with TRIAL_AD_CREDITS if it
    doesn't exist yet. Idempotent — race-safe: two near-simultaneous calls
    for the same brand-new user can both see no existing row and both
    attempt the insert; the loser hits a 23505 unique-violation on
    owner_id, which just means the row now exists either way, not a real
    failure. Any other error still propagates."""
    existing = with_retry(lambda: supabase.table("ad_credits")
        .select("owner_id")
        .eq("owner_id", user_id)
        .execute())
    existing = ensure_supabase_response(existing, "check ad credits row")
    if not existing.data:
        def _insert():
            # Swallowed here, inside the retried closure, so a 23505 (lost
            # the race, row already exists) short-circuits on the first
            # attempt instead of burning with_retry's full backoff
            # retrying an insert that can never succeed.
            try:
                return supabase.table("ad_credits").insert({
                    "owner_id": user_id,
                    "credits": TRIAL_AD_CREDITS,
                }).execute()
            except APIError as e:
                if e.code != "23505":
                    raise
        with_retry(_insert)


def _get_ad_credits(user_id: str) -> int:
    _ensure_ad_credits_row(user_id)
    res = with_retry(lambda: supabase.table("ad_credits")
        .select("credits")
        .eq("owner_id", user_id)
        .execute())
    res = ensure_supabase_response(res, "get ad credits")
    return res.data[0]["credits"] if res.data else 0


def _spend_ad_credit(user_id: str) -> int:
    """Checks the caller has at least 1 credit, decrements by 1, and
    returns the new balance. Raises 402 if there's nothing left to spend.
    One shared helper for the three routes that each cost a credit,
    instead of duplicating the same check/decrement block three times."""
    credits = _get_ad_credits(user_id)
    if credits <= 0:
        raise HTTPException(
            status_code=402,
            detail="You're out of ad credits. Upgrade to keep generating.",
        )
    new_credits = credits - 1
    supabase.table("ad_credits").update({
        "credits": new_credits,
        "updated_at": "now()",
    }).eq("owner_id", user_id).execute()
    return new_credits


@app.get("/ads/credits", response_model=AdCreditsOut, tags=["ads"])
def ad_credits(user_id: str = Depends(get_current_user_id)):
    try:
        return {"credits": _get_ad_credits(user_id)}
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# BUSINESS PROFILE
# -----------------------
def _get_business_category(user_id: str) -> str:
    res = with_retry(lambda: supabase.table("business_profile")
        .select("category")
        .eq("owner_id", user_id)
        .execute())
    res = ensure_supabase_response(res, "get business profile")
    if res.data:
        return res.data[0]["category"]
    return "other"


def _set_business_category(user_id: str, category: str) -> None:
    with_retry(lambda: supabase.table("business_profile").upsert({
        "owner_id": user_id,
        "category": category,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, on_conflict="owner_id").execute())


@app.get("/business-profile", response_model=BusinessProfileOut, tags=["ads"])
def get_business_profile(user_id: str = Depends(get_current_user_id)):
    try:
        return {"category": _get_business_category(user_id)}
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/business-profile", response_model=BusinessProfileOut, tags=["ads"])
def set_business_profile(req: SetBusinessProfileRequest, user_id: str = Depends(get_current_user_id)):
    try:
        _set_business_category(user_id, req.category)
        return {"category": req.category}
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# AD GENERATION: caption copy + banner image
# -----------------------
# Considered-purchase categories (real_estate, automotive, home_services,
# education_coaching, professional_services) lean on trust/detail rather
# than the discount/urgency framing that works for everyday-purchase
# categories (retail, restaurant_cafe, ecommerce).
CONTENT_PLAN_CATEGORY_GUIDANCE = {
    "retail": "This is a retail/product shop — emphasize new arrivals, everyday value, and product quality when writing post ideas.",
    "restaurant_cafe": "This is a restaurant/cafe — emphasize taste, fresh food, and daily specials/offers when writing post ideas.",
    "health_beauty": "This is a health/beauty business (salon, spa, clinic) — emphasize service quality, expertise, hygiene, and results when writing post ideas.",
    "professional_services": "This is a professional services business (legal, accounting, consulting, freelance) — emphasize expertise, credentials, and client trust when writing post ideas.",
    "home_services": "This is a home services business (interior design, renovation, furniture, repair) — emphasize past work/portfolio, craftsmanship, and personalized consultation when writing post ideas.",
    "real_estate": "This is a real estate business — emphasize location, transparent paperwork, and viewing opportunities when writing post ideas.",
    "automotive": "This is an automotive business (vehicle sales or service) — emphasize vehicle condition, trustworthiness, and test-drive/viewing opportunities when writing post ideas.",
    "education_coaching": "This is an education/coaching business — emphasize results, reputation, experienced instructors, and structured curriculum when writing post ideas.",
    "fitness_sports": "This is a fitness/sports business (gym, studio, trainer) — emphasize real results, community, and expert coaching when writing post ideas.",
    "events_entertainment": "This is an events/entertainment business — emphasize the experience, atmosphere, and booking availability when writing post ideas.",
    "ecommerce": "This is an online/ecommerce business — emphasize delivery, showing off multiple products, and limited-stock urgency when writing post ideas.",
    "technology_software": "This is a technology/software business — emphasize reliability, key features/benefits, and support quality when writing post ideas.",
    "other": "Write generally effective post ideas for this small business.",
}


def _generate_ad_copy(item_description: str, category: str = "other") -> list[dict]:
    """Three distinct Facebook caption + WhatsApp-variant pairs, so the
    business owner picks a favorite instead of getting stuck with whatever
    the model wrote first. Text generation is cheap (unlike the banner
    image below), so all 3 are bundled into the same 1-credit generation —
    kept completely separate from the image call, nothing about this needs
    Gemini."""
    category_guidance = CONTENT_PLAN_CATEGORY_GUIDANCE.get(category, CONTENT_PLAN_CATEGORY_GUIDANCE["other"])
    prompt = f"""You are a Facebook ad copywriter for small businesses.

{category_guidance}

From the product/offer description below, write 3 distinct versions of ad copy — each in a different tone/angle (e.g. one straightforwardly informational, one emotion/benefit-focused, one urgency/offer-focused) so the business owner can pick their favorite. Don't just reword the same thing — each version needs a genuinely different approach.

Each version has two parts:
1. "facebook_caption" — a short, engaging Facebook post caption (2-3 lines, use emoji where appropriate, written to attract local customers)
2. "whatsapp_message" — an even shorter WhatsApp message version (1-2 lines, for sending directly to a customer)

Keep any numbers/prices exactly as given — don't invent new prices or offers.

Product/offer description: {item_description}

Respond with ONLY this JSON format, nothing else:
{{"captions": [{{"facebook_caption": "", "whatsapp_message": ""}}, {{"facebook_caption": "", "whatsapp_message": ""}}, {{"facebook_caption": "", "whatsapp_message": ""}}]}}
"""
    response = with_retry(
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write short, appealing marketing copy for small business owners."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        ),
        exceptions=RETRYABLE_OPENAI_ERRORS,
    )
    ai_text = response.choices[0].message.content.strip()
    m = re.search(r"```(?:json)?\n(.*?)```", ai_text, re.S)
    ai_text_clean = m.group(1).strip() if m else ai_text.strip().strip("`").strip()
    parsed = json.loads(ai_text_clean)
    captions = parsed.get("captions") or []
    return [
        {
            "facebook_caption": c.get("facebook_caption", ""),
            "whatsapp_message": c.get("whatsapp_message", ""),
        }
        for c in captions[:3]
    ] or [{"facebook_caption": "", "whatsapp_message": ""}]


def _generate_banner_image(image_bytes: bytes, mime_type: str, item_description: str) -> bytes:
    """Edits the user's own photo (background only) via Gemini —
    deliberately does NOT ask the model to add any text to the image. Text
    rendered by image models is unreliable/garbled; the caller overlays
    real text on this background separately."""
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="AI image generation isn't enabled yet.")

    prompt = (
        "Edit this product photo into a clean, professional promotional banner "
        "background suitable for a Facebook ad for a small business. "
        "Keep the actual product in the photo exactly as it is — do not change, "
        "replace, warp, or redraw the product itself. Only improve or replace the "
        "background: make it clean, well-lit, and visually appealing, contextually "
        f"fitting for this item/offer: {item_description}. "
        "Do NOT add any text, letters, numbers, or words anywhere in the image — "
        "leave clean, uncluttered space (e.g. near the top or bottom) where text "
        "will be added afterward by a separate step. Output a high-quality square image."
    )
    response = with_retry(
        lambda: gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
        ),
        exceptions=(Exception,),
        attempts=2,
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    raise Exception("Gemini did not return an image")


def _generate_ai_banner_image(item_description: str, category: str = "other") -> bytes:
    """Generates a banner image from scratch (no real photo) for users
    without one to upload. The pictured product is AI-imagined rather than
    the business's actual item, so this only ever runs when the frontend
    explicitly sends no file — never a silent fallback for a failed upload."""
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="AI image generation isn't enabled yet.")

    category_guidance = CONTENT_PLAN_CATEGORY_GUIDANCE.get(category, CONTENT_PLAN_CATEGORY_GUIDANCE["other"])
    prompt = (
        "Generate a clean, professional, photorealistic promotional banner image "
        "for a Facebook ad for a small business. "
        f"Context: {category_guidance} "
        f"The image should visually represent this product/offer: {item_description}. "
        "Make it well-lit, visually appealing, and contextually appropriate. "
        "Do NOT add any text, letters, numbers, or words anywhere in the image — "
        "leave clean, uncluttered space (e.g. near the top or bottom) where text "
        "will be added afterward by a separate step. Output a high-quality square image."
    )
    response = with_retry(
        lambda: gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        ),
        exceptions=(Exception,),
        attempts=2,
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    raise Exception("Gemini did not return an image")


async def _get_banner_image(image_bytes: Optional[bytes], mime_type: Optional[str], item_description: str, category: str) -> bytes:
    # Gemini image generation is a blocking call and can take well over a
    # minute (especially generating from scratch, no reference photo) — run
    # it off the event loop so a slow generation doesn't stall every other
    # request this worker is handling, health checks included. (This fixes
    # a real bug: without run_in_threadpool, one slow generation freezes
    # the whole backend for every user until it finishes.)
    if image_bytes:
        return await run_in_threadpool(_generate_banner_image, image_bytes, mime_type or "image/jpeg", item_description)
    return await run_in_threadpool(_generate_ai_banner_image, item_description, category)


@app.post("/ads/generate", response_model=AdGenerateResponse, tags=["ads"])
@limiter.limit("10/minute")
async def generate_ad(
    request: Request,
    item_description: str,
    file: Optional[UploadFile] = File(None),
    user_id: str = Depends(get_current_user_id),
):
    try:
        credits = _get_ad_credits(user_id)
        if credits <= 0:
            raise HTTPException(
                status_code=402,
                detail="You're out of ad credits. Upgrade to keep generating.",
            )

        image_bytes = await file.read() if file is not None else None
        mime_type = file.content_type if file is not None else None
        category = _get_business_category(user_id)

        copy = _generate_ad_copy(item_description, category)
        banner_bytes = await _get_banner_image(image_bytes, mime_type, item_description, category)

        new_credits = _spend_ad_credit(user_id)

        return {
            "captions": copy,
            "banner_image_base64": base64.b64encode(banner_bytes).decode("ascii"),
            "credits_remaining": new_credits,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ads/generate-image-variant", response_model=AdImageVariantResponse, tags=["ads"])
@limiter.limit("10/minute")
async def generate_ad_image_variant(
    request: Request,
    item_description: str,
    file: Optional[UploadFile] = File(None),
    user_id: str = Depends(get_current_user_id),
):
    """A second (or third) banner background option for the same post —
    unlike the bundled-free caption variants, each extra image costs its
    own credit since Gemini image generation is the expensive part of a
    generation, not the text."""
    try:
        credits = _get_ad_credits(user_id)
        if credits <= 0:
            raise HTTPException(
                status_code=402,
                detail="You're out of ad credits. Upgrade to keep generating.",
            )

        image_bytes = await file.read() if file is not None else None
        mime_type = file.content_type if file is not None else None
        category = _get_business_category(user_id)
        banner_bytes = await _get_banner_image(image_bytes, mime_type, item_description, category)

        new_credits = _spend_ad_credit(user_id)

        return {
            "banner_image_base64": base64.b64encode(banner_bytes).decode("ascii"),
            "credits_remaining": new_credits,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# WEEKLY CONTENT PLAN
# -----------------------
CONTENT_PLAN_THEMES = {"restock", "popular", "offer", "general"}


def _generate_manual_content_plan_ideas(input_text: str, category: str) -> list[dict]:
    """Grounded in a short user-typed description of what they want to
    promote this week (no inventory/sales data — this app has no such
    ledger)."""
    prompt = f"""You are a social media content planner for a small business.

{CONTENT_PLAN_CATEGORY_GUIDANCE.get(category, CONTENT_PLAN_CATEGORY_GUIDANCE["other"])}

What the business owner wants to post about this week: {input_text}

Using the description above, plan 5 Facebook/social media post ideas for the week (Monday to Friday, day codes: Mon, Tue, Wed, Thu, Fri). Each idea should take a different angle on the description above — don't invent new facts/prices, only use what's in the description. Each post's theme must be exactly one of these 4 values, no others: "restock" (something new/back in stock), "popular" (popular/high demand), "offer" (special offer/terms), or "general" (general promotion).

The 5 posts shouldn't feel like 5 disconnected posts — the whole week should feel like one connected campaign. Keep a natural flow: build interest early in the week (Mon/Tue, theme: restock/popular), talk about trust or quality in the middle (Wed, theme: general), and drive action with an offer or time-limited urgency by the end (Thu/Fri, theme: offer). Vary the angle each day — don't repeat the same point.

Respond with ONLY this JSON format, nothing else:
{{"posts": [{{"day": "Mon", "theme": "restock", "idea_text": "one-line post idea", "source_items": [], "media_type": "image"}}]}}
"""
    response = with_retry(
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You plan a week of social media posts for a small business based on what the owner says they want to promote."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        ),
        exceptions=RETRYABLE_OPENAI_ERRORS,
    )
    ai_text = response.choices[0].message.content.strip()
    m = re.search(r"```(?:json)?\n(.*?)```", ai_text, re.S)
    ai_text_clean = m.group(1).strip() if m else ai_text.strip().strip("`").strip()
    parsed = json.loads(ai_text_clean)
    ideas = parsed.get("posts") or []

    return [
        {
            "day": idea.get("day", ""),
            # Clamped, not trusted as-is — the frontend only knows these 4
            # theme keys.
            "theme": idea.get("theme") if idea.get("theme") in CONTENT_PLAN_THEMES else "general",
            "idea_text": idea.get("idea_text", ""),
            "source_items": idea.get("source_items") or [],
            "media_type": idea.get("media_type", "image"),
            "status": "idea",
            "caption": None,
            "whatsapp_message": None,
            "image_base64": None,
        }
        for idea in ideas
    ]


@app.post("/content-plan/generate", response_model=ContentPlanOut, tags=["ads"])
@limiter.limit("5/minute")
def generate_content_plan(
    request: Request,
    req: GenerateContentPlanRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Free — this only writes text post ideas, no image generation. Each
    idea only costs a credit once the user actually taps to generate that
    specific day's real caption+banner, same as /ads/generate."""
    try:
        category = _get_business_category(user_id)
        input_text = (req.input_text or "").strip()
        if not input_text:
            raise HTTPException(status_code=400, detail="Tell us what you want to post about this week.")
        posts = _generate_manual_content_plan_ideas(input_text, category)

        today = datetime.now(timezone.utc).date()
        period_end = today + timedelta(days=6)

        # Regenerating replaces the user's one current plan — /content-plan/current
        # only orders by period_start, so leaving the old row behind risks it
        # resurfacing on a period_start tie instead of the plan just generated.
        with_retry(lambda: supabase.table("content_plans").delete().eq("owner_id", user_id).execute())

        insert_res = with_retry(lambda: supabase.table("content_plans").insert({
            "owner_id": user_id,
            "period_start": today.isoformat(),
            "period_end": period_end.isoformat(),
            "status": "active",
            "posts": posts,
            "input_text": input_text,
        }).execute())
        insert_res = ensure_supabase_response(insert_res, "insert content plan")
        row = insert_res.data[0]
        return {
            "id": row["id"],
            "period_start": row["period_start"],
            "period_end": row["period_end"],
            "status": row["status"],
            "posts": row["posts"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/content-plan/current", tags=["ads"])
def get_current_content_plan(user_id: str = Depends(get_current_user_id)):
    try:
        res = with_retry(lambda: supabase.table("content_plans")
            .select("*")
            .eq("owner_id", user_id)
            .order("period_start", desc=True)
            .limit(1)
            .execute())
        res = ensure_supabase_response(res, "get current content plan")
        if not res.data:
            return None
        row = res.data[0]
        return {
            "id": row["id"],
            "period_start": row["period_start"],
            "period_end": row["period_end"],
            "status": row["status"],
            "posts": row["posts"],
        }
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/content-plan/{plan_id}/posts/{day}/generate", response_model=AdGenerateResponse, tags=["ads"])
@limiter.limit("10/minute")
async def generate_content_plan_post(
    request: Request,
    plan_id: str,
    day: str,
    file: Optional[UploadFile] = File(None),
    idea_text: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user_id),
):
    """Generates the real caption+banner for one specific day's idea in an
    existing plan — reuses _generate_ad_copy/_get_banner_image unchanged
    (same as /ads/generate), just sources item_description from that
    day's plan idea instead of a freely-typed description.

    idea_text, if provided, overrides the plan's stored suggestion — lets
    the user edit the AI's idea before spending a credit on it, without
    needing a separate "update plan" round-trip."""
    try:
        credits = _get_ad_credits(user_id)
        if credits <= 0:
            raise HTTPException(
                status_code=402,
                detail="You're out of ad credits. Upgrade to keep generating.",
            )

        plan_res = with_retry(lambda: supabase.table("content_plans")
            .select("*")
            .eq("id", plan_id)
            .eq("owner_id", user_id)
            .execute())
        plan_res = ensure_supabase_response(plan_res, "get content plan for post generation")
        if not plan_res.data:
            raise HTTPException(status_code=404, detail="Plan not found.")
        plan_row = plan_res.data[0]

        posts = plan_row["posts"]
        post_index = next((i for i, p in enumerate(posts) if p["day"] == day), None)
        if post_index is None:
            raise HTTPException(status_code=404, detail="No idea found for that day.")

        item_description = (idea_text or "").strip() or posts[post_index]["idea_text"]

        image_bytes = await file.read() if file is not None else None
        mime_type = file.content_type if file is not None else None
        category = _get_business_category(user_id)

        copy = _generate_ad_copy(item_description, category)
        banner_bytes = await _get_banner_image(image_bytes, mime_type, item_description, category)
        banner_b64 = base64.b64encode(banner_bytes).decode("ascii")

        # Persist the first caption variant as the default so a later
        # revisit still shows something real — the frontend calls /select
        # below if the user picks a different variant.
        posts[post_index] = {
            **posts[post_index],
            "idea_text": item_description,
            "status": "generated",
            "caption": copy[0]["facebook_caption"],
            "whatsapp_message": copy[0]["whatsapp_message"],
            "image_base64": banner_b64,
        }

        new_credits = _spend_ad_credit(user_id)
        supabase.table("content_plans").update({"posts": posts}).eq("id", plan_id).execute()

        return {
            "captions": copy,
            "banner_image_base64": banner_b64,
            "credits_remaining": new_credits,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/content-plan/{plan_id}/posts/{day}/select", tags=["ads"])
def select_content_plan_post(
    plan_id: str,
    day: str,
    req: SelectContentPlanPostRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Records which caption/image variant the user actually picked — no
    AI call, no credit cost — so a later revisit shows their choice
    instead of always falling back to the first generated variant."""
    try:
        plan_res = with_retry(lambda: supabase.table("content_plans")
            .select("*")
            .eq("id", plan_id)
            .eq("owner_id", user_id)
            .execute())
        plan_res = ensure_supabase_response(plan_res, "get content plan for post selection")
        if not plan_res.data:
            raise HTTPException(status_code=404, detail="Plan not found.")
        plan_row = plan_res.data[0]

        posts = plan_row["posts"]
        post_index = next((i for i, p in enumerate(posts) if p["day"] == day), None)
        if post_index is None:
            raise HTTPException(status_code=404, detail="No idea found for that day.")

        posts[post_index] = {
            **posts[post_index],
            "caption": req.caption,
            "whatsapp_message": req.whatsapp_message,
            **({"image_base64": req.image_base64} if req.image_base64 is not None else {}),
        }
        supabase.table("content_plans").update({"posts": posts}).eq("id", plan_id).execute()
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
