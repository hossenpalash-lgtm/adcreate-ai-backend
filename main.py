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
import ipaddress
import socket
import subprocess
import tempfile
import imageio_ffmpeg
import csv
import io
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone

# -----------------------
# LOGGING
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("punqle")

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
    title="Punqle Backend",
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


class ReferralRedeemRequest(BaseModel):
    referrer_id: str


class ReferralRedeemResponse(BaseModel):
    granted: bool
    credits_remaining: int


class ReferralStatusOut(BaseModel):
    referral_code: str
    successful_referrals: int
    max_referrals: int


class GenerateVideoRequest(BaseModel):
    item_description: str
    image_base64: Optional[str] = None
    image_mime_type: Optional[str] = None
    aspect_ratio: str = "16:9"

    @field_validator("aspect_ratio")
    @classmethod
    def validate_video_aspect_ratio(cls, v):
        if v not in ("16:9", "9:16"):
            raise ValueError("aspect_ratio must be '16:9' or '9:16'")
        return v


class VideoOperationOut(BaseModel):
    operation: dict
    headline: str


class VideoStatusRequest(BaseModel):
    operation: dict
    # Whatever the client currently has — the user can edit the AI-suggested
    # headline any time during the 1-2 minute generation wait, and whatever
    # it is at poll time is what gets burned onto the finished video. Empty
    # string means the user cleared it deliberately — skip burn-in entirely.
    headline: str = ""


class VideoStatusResponse(BaseModel):
    done: bool
    video_base64: Optional[str] = None
    credits_remaining: Optional[int] = None


class ImportedProductOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    price: Optional[str] = None
    image_url: Optional[str] = None


class ImportedProductsListResponse(BaseModel):
    products: list[ImportedProductOut]


class FetchProductImageRequest(BaseModel):
    url: str


class FetchProductImageResponse(BaseModel):
    image_base64: str
    mime_type: str


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


class TranslateCaptionsRequest(BaseModel):
    captions: list[AdCaptionVariant]
    target_language: str


class TranslateCaptionsResponse(BaseModel):
    captions: list[AdCaptionVariant]


class FetchProductLinkRequest(BaseModel):
    url: str


class FetchProductLinkResponse(BaseModel):
    title: str
    description: str
    image_base64: Optional[str] = None
    mime_type: Optional[str] = None


BUSINESS_CATEGORIES = {
    "retail", "restaurant_cafe", "health_beauty", "professional_services",
    "home_services", "real_estate", "automotive", "education_coaching",
    "fitness_sports", "events_entertainment", "ecommerce",
    "technology_software", "other",
}


class BusinessProfileOut(BaseModel):
    category: str
    brand_color: Optional[str] = None
    logo_base64: Optional[str] = None
    logo_mime_type: Optional[str] = None
    brand_name: Optional[str] = None


class SetBusinessProfileRequest(BaseModel):
    # All optional — this is a partial update. Omitted fields keep
    # whatever's already saved rather than being cleared to null, so the
    # Brand Kit panel (color+logo) and the Weekly Plan category picker
    # can each save independently without clobbering the other.
    category: Optional[str] = None
    brand_color: Optional[str] = None
    logo_base64: Optional[str] = None
    logo_mime_type: Optional[str] = None
    brand_name: Optional[str] = None

    @field_validator("brand_name")
    @classmethod
    def validate_brand_name(cls, v):
        if v is not None and len(v) > 60:
            raise ValueError("Business name is too long (max 60 characters).")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        if v is not None and v not in BUSINESS_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(BUSINESS_CATEGORIES)}")
        return v

    @field_validator("brand_color")
    @classmethod
    def validate_brand_color(cls, v):
        if v is not None and not re.fullmatch(r"#[0-9A-Fa-f]{6}", v):
            raise ValueError("brand_color must be a hex color like #3B5BFF")
        return v

    @field_validator("logo_base64")
    @classmethod
    def validate_logo_size(cls, v):
        # ~2MB decoded (base64 is ~4/3 the size of the raw bytes) — a
        # logo has no business being bigger than this, and it's stored
        # as a plain text column rather than object storage.
        if v is not None and len(v) > 2_800_000:
            raise ValueError("Logo image is too large (max ~2MB).")
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


class StockPhotoResult(BaseModel):
    id: str
    thumbnail_url: str
    full_url: str
    photographer: str


class StockPhotoSearchResponse(BaseModel):
    results: list[StockPhotoResult]


class FetchStockPhotoRequest(BaseModel):
    url: str


class FetchStockPhotoResponse(BaseModel):
    image_base64: str
    mime_type: str


class GeneratedPostOut(BaseModel):
    id: str
    item_description: str
    facebook_caption: str
    whatsapp_message: str
    image_base64: str
    created_at: str


class GeneratedPostHistoryResponse(BaseModel):
    posts: list[GeneratedPostOut]


class SuggestHashtagsRequest(BaseModel):
    item_description: str


class SuggestHashtagsResponse(BaseModel):
    hashtags: list[str]


class IdeaLabsResponse(BaseModel):
    ideas: list[str]


class BlogToPostsRequest(BaseModel):
    url: str


class BlogToPostsResponse(BaseModel):
    title: str
    ideas: list[str]


class CompetitorAnalysisRequest(BaseModel):
    url: str


class CompetitorAnalysisResponse(BaseModel):
    competitor_name: str
    summary: str
    differentiation_ideas: list[str]


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
# Optional — only stock-photo search needs this, same reasoning as above.
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()

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
    return {"status": "Punqle backend is running"}


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


def _spend_ad_credits(user_id: str, amount: int) -> int:
    """Like _spend_ad_credit but for actions costing more than 1 (video) —
    kept separate rather than generalizing the 1-credit callers onto this,
    since those are already correct and this is only new for video."""
    credits = _get_ad_credits(user_id)
    if credits < amount:
        raise HTTPException(
            status_code=402,
            detail=f"This needs {amount} credits — you have {credits}. Upgrade to keep generating.",
        )
    new_credits = credits - amount
    with_retry(lambda: supabase.table("ad_credits").update({
        "credits": new_credits,
        "updated_at": "now()",
    }).eq("owner_id", user_id).execute())
    return new_credits


@app.get("/ads/credits", response_model=AdCreditsOut, tags=["ads"])
def ad_credits(user_id: str = Depends(get_current_user_id)):
    try:
        return {"credits": _get_ad_credits(user_id)}
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _add_ad_credits(user_id: str, amount: int) -> int:
    """Grants bonus credits (referrals, etc.) rather than spending them —
    the mirror of _spend_ad_credit. Ensures the row exists first, same
    lazy-creation as everywhere else credits are touched."""
    credits = _get_ad_credits(user_id)
    new_credits = credits + amount
    with_retry(lambda: supabase.table("ad_credits").update({
        "credits": new_credits,
        "updated_at": "now()",
    }).eq("owner_id", user_id).execute())
    return new_credits


# -----------------------
# REFERRALS
# -----------------------
# Both sides get a bonus once a referred account is confirmed real (its
# own JWT-verified user_id, via get_current_user_id) — capped per referrer
# to bound the cost of someone farming fake accounts against their own
# code, same lesson as the sister app's voice-command referral bonus cap.
REFERRAL_BONUS_CREDITS = 5
REFERRAL_MAX_SUCCESSFUL = 5


def _is_real_user(user_id: str) -> bool:
    """A referrer_id arrives as a plain string in the request body, not
    something JWT-verified like the caller's own id — confirm it actually
    corresponds to a Supabase auth user before crediting it, rather than
    silently creating a stray ad_credits row for a typo'd or made-up id."""
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=10,
        )
    except requests.RequestException:
        return False
    return resp.status_code == 200


@app.post("/referral/redeem", response_model=ReferralRedeemResponse, tags=["referral"])
@limiter.limit("10/minute")
def redeem_referral(
    request: Request,
    req: ReferralRedeemRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Free — grants REFERRAL_BONUS_CREDITS to both sides, once per new
    account (referee_id is unique, so a reload/retry after a successful
    redemption is a harmless no-op rather than a double-grant)."""
    try:
        referrer_id = (req.referrer_id or "").strip()
        if not referrer_id or referrer_id == user_id:
            return {"granted": False, "credits_remaining": _get_ad_credits(user_id)}

        already = with_retry(lambda: supabase.table("referrals")
            .select("id")
            .eq("referee_id", user_id)
            .execute())
        already = ensure_supabase_response(already, "check referral redemption")
        if already.data:
            return {"granted": False, "credits_remaining": _get_ad_credits(user_id)}

        if not _is_real_user(referrer_id):
            raise HTTPException(status_code=400, detail="That referral link isn't valid.")

        count_res = with_retry(lambda: supabase.table("referrals")
            .select("id", count="exact")
            .eq("referrer_id", referrer_id)
            .execute())
        successful = count_res.count or 0
        if successful >= REFERRAL_MAX_SUCCESSFUL:
            # Referrer's already hit the cap — don't record this attempt,
            # so a later real slot (if the referrer's count could ever
            # drop) isn't blocked by a stale row, and the referee simply
            # doesn't get a bonus for an already-capped code.
            return {"granted": False, "credits_remaining": _get_ad_credits(user_id)}

        try:
            with_retry(lambda: supabase.table("referrals").insert({
                "referrer_id": referrer_id,
                "referee_id": user_id,
            }).execute())
        except APIError as e:
            if e.code == "23505":
                # Lost a race with a concurrent redemption for this same
                # referee — already recorded, not a real failure.
                return {"granted": False, "credits_remaining": _get_ad_credits(user_id)}
            raise

        new_credits = _add_ad_credits(user_id, REFERRAL_BONUS_CREDITS)
        _add_ad_credits(referrer_id, REFERRAL_BONUS_CREDITS)
        return {"granted": True, "credits_remaining": new_credits}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/referral/status", response_model=ReferralStatusOut, tags=["referral"])
def referral_status(user_id: str = Depends(get_current_user_id)):
    try:
        count_res = with_retry(lambda: supabase.table("referrals")
            .select("id", count="exact")
            .eq("referrer_id", user_id)
            .execute())
        return {
            "referral_code": user_id,
            "successful_referrals": count_res.count or 0,
            "max_referrals": REFERRAL_MAX_SUCCESSFUL,
        }
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# CONTENT LIBRARY (generation history)
# -----------------------
GENERATED_POSTS_RETENTION = 20


def _save_generated_post(user_id: str, item_description: str, caption: dict, image_base64: str) -> None:
    """Best-effort — a save failure shouldn't break the actual generation
    the user is waiting on, so this never raises. Retention capped
    per-user (unlike Brand Kit's one-row-per-user, this table grows
    unbounded with usage) so it can't quietly fill up the database over
    a user's lifetime."""
    try:
        with_retry(lambda: supabase.table("generated_posts").insert({
            "owner_id": user_id,
            "item_description": item_description,
            "facebook_caption": caption.get("facebook_caption", ""),
            "whatsapp_message": caption.get("whatsapp_message", ""),
            "image_base64": image_base64,
        }).execute())
        existing = with_retry(lambda: supabase.table("generated_posts")
            .select("id")
            .eq("owner_id", user_id)
            .order("created_at", desc=True)
            .execute())
        existing = ensure_supabase_response(existing, "list generated posts for retention")
        stale_ids = [row["id"] for row in existing.data[GENERATED_POSTS_RETENTION:]]
        if stale_ids:
            supabase.table("generated_posts").delete().in_("id", stale_ids).execute()
    except Exception as e:
        logger.error("Failed to save generated post to history: %s", str(e), exc_info=True)


@app.get("/ads/history", response_model=GeneratedPostHistoryResponse, tags=["ads"])
def get_history(user_id: str = Depends(get_current_user_id)):
    try:
        res = with_retry(lambda: supabase.table("generated_posts")
            .select("id, item_description, facebook_caption, whatsapp_message, image_base64, created_at")
            .eq("owner_id", user_id)
            .order("created_at", desc=True)
            .limit(GENERATED_POSTS_RETENTION)
            .execute())
        res = ensure_supabase_response(res, "get generated post history")
        return {"posts": res.data}
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ads/history/{post_id}", tags=["ads"])
def delete_history_post(post_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        # Filtering by owner_id too (not just id) is the actual access
        # check here — this backend talks to Supabase with a service-role
        # key, so nothing stops a request for someone else's row at the
        # DB layer without this.
        supabase.table("generated_posts").delete().eq("id", post_id).eq("owner_id", user_id).execute()
        return {"deleted": True}
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


def _get_business_profile(user_id: str) -> dict:
    res = with_retry(lambda: supabase.table("business_profile")
        .select("category, brand_color, logo_base64, logo_mime_type, brand_name")
        .eq("owner_id", user_id)
        .execute())
    res = ensure_supabase_response(res, "get business profile")
    if res.data:
        row = res.data[0]
        return {
            "category": row.get("category") or "other",
            "brand_color": row.get("brand_color"),
            "logo_base64": row.get("logo_base64"),
            "logo_mime_type": row.get("logo_mime_type"),
            "brand_name": row.get("brand_name"),
        }
    return {"category": "other", "brand_color": None, "logo_base64": None, "logo_mime_type": None, "brand_name": None}


def _update_business_profile(
    user_id: str,
    category: Optional[str] = None,
    brand_color: Optional[str] = None,
    logo_base64: Optional[str] = None,
    logo_mime_type: Optional[str] = None,
    brand_name: Optional[str] = None,
) -> dict:
    """Fetch-then-merge partial update — each field is only overwritten
    if the caller actually provided it, so the Brand Kit panel (color +
    logo + name) and the Weekly Plan category picker don't stomp on each
    other's fields when saving independently."""
    current = _get_business_profile(user_id)
    payload = {
        "owner_id": user_id,
        "category": category if category is not None else current["category"],
        "brand_color": brand_color if brand_color is not None else current["brand_color"],
        "logo_base64": logo_base64 if logo_base64 is not None else current["logo_base64"],
        "logo_mime_type": logo_mime_type if logo_mime_type is not None else current["logo_mime_type"],
        "brand_name": brand_name if brand_name is not None else current["brand_name"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with_retry(lambda: supabase.table("business_profile").upsert(payload, on_conflict="owner_id").execute())
    return {k: payload[k] for k in ("category", "brand_color", "logo_base64", "logo_mime_type", "brand_name")}


@app.get("/business-profile", response_model=BusinessProfileOut, tags=["ads"])
def get_business_profile(user_id: str = Depends(get_current_user_id)):
    try:
        return _get_business_profile(user_id)
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/business-profile", response_model=BusinessProfileOut, tags=["ads"])
def set_business_profile(req: SetBusinessProfileRequest, user_id: str = Depends(get_current_user_id)):
    try:
        return _update_business_profile(
            user_id,
            category=req.category,
            brand_color=req.brand_color,
            logo_base64=req.logo_base64,
            logo_mime_type=req.logo_mime_type,
            brand_name=req.brand_name,
        )
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


def _suggest_hashtags(item_description: str, category: str) -> list[str]:
    """Free — text-only, same economics as translate-captions. Separate
    from _generate_ad_copy's caption text on purpose: Predis's own
    tutorial has a dedicated hashtag-picker step distinct from the
    caption, letting the user toggle individual tags rather than getting
    them locked into whatever the AI wrote inline."""
    category_guidance = CONTENT_PLAN_CATEGORY_GUIDANCE.get(category, CONTENT_PLAN_CATEGORY_GUIDANCE["other"])
    prompt = f"""Suggest 12 relevant hashtags for a small business Facebook/Instagram post. Mix a few broad/popular tags with several niche/specific ones. Don't include the # symbol, no spaces within a tag.

{category_guidance}

The post is about: {item_description}

Respond with ONLY this JSON format, nothing else:
{{"hashtags": ["tag1", "tag2"]}}
"""
    response = with_retry(
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You suggest relevant social media hashtags for small business marketing."},
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
    tags = parsed.get("hashtags") or []
    return [str(t).lstrip("#").strip() for t in tags if str(t).strip()][:15]


def _generate_idea_labs_ideas(category: str) -> list[str]:
    """Free — text-only GPT call. For a user with no specific product in
    mind yet: general post ideas grounded only in their saved business
    category, not any real inventory/sales data (this app has none)."""
    category_guidance = CONTENT_PLAN_CATEGORY_GUIDANCE.get(category, CONTENT_PLAN_CATEGORY_GUIDANCE["other"])
    prompt = f"""You are a social media content strategist for a small business.

{category_guidance}

The business owner has no specific product or offer in mind right now — they just want ideas for what to post about next. Suggest 8 distinct, concrete post ideas a small business in this category could realistically write today. Don't invent fake stats, prices, or specific products you can't know about — keep ideas general enough to apply, but concrete and actionable, e.g. "Show a before/after of your most requested service" rather than vague advice like "engage your audience."

Respond with ONLY this JSON format, nothing else:
{{"ideas": ["idea 1", "idea 2"]}}
"""
    response = with_retry(
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You give small business owners concrete social media post ideas when they don't have a specific product in mind yet."},
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
    ideas = parsed.get("ideas") or []
    return [str(i).strip() for i in ideas if str(i).strip()][:10]


def _translate_captions(captions: list[dict], target_language: str) -> list[dict]:
    """Text-only, reuses the already-generated captions instead of
    regenerating from scratch — cheap enough (gpt-4o-mini, small prompt)
    to bundle free rather than spend a credit on it."""
    prompt = f"""Translate the following Facebook ad captions and WhatsApp messages into {target_language}. Keep the same tone and meaning, adapt naturally rather than translating word-for-word, and keep any numbers/prices exactly as given.

Captions to translate:
{json.dumps(captions)}

Respond with ONLY this JSON format, nothing else, same number of items in the same order:
{{"captions": [{{"facebook_caption": "", "whatsapp_message": ""}}]}}
"""
    response = with_retry(
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional marketing translator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        ),
        exceptions=RETRYABLE_OPENAI_ERRORS,
    )
    ai_text = response.choices[0].message.content.strip()
    m = re.search(r"```(?:json)?\n(.*?)```", ai_text, re.S)
    ai_text_clean = m.group(1).strip() if m else ai_text.strip().strip("`").strip()
    parsed = json.loads(ai_text_clean)
    translated = parsed.get("captions") or []
    return [
        {
            "facebook_caption": c.get("facebook_caption", ""),
            "whatsapp_message": c.get("whatsapp_message", ""),
        }
        for c in translated
    ]


_MAX_LINK_FETCH_BYTES = 3_000_000  # generous for a product page/photo, small enough to bound abuse


def _assert_public_url(url: str) -> None:
    """SSRF guard — this endpoint fetches a URL the user typed in, so
    without this check a request could be pointed at internal services
    or a cloud metadata endpoint (e.g. 169.254.169.254) that Render's
    network can reach but the public internet can't. Best-effort, not
    exhaustive: resolves once and checks the IP is globally routable,
    doesn't defend against DNS-rebinding between this check and the
    actual request — acceptable for this app's current scale/threat
    model, not a substitute for a proper egress proxy if that changes."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Please enter a valid product page link.")
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Please enter a valid product page link.")
    try:
        resolved_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        raise HTTPException(status_code=400, detail="Couldn't resolve that link.")
    if not ipaddress.ip_address(resolved_ip).is_global:
        raise HTTPException(status_code=400, detail="That link isn't allowed.")


def _fetch_url_bytes(url: str) -> tuple[bytes, str]:
    _assert_public_url(url)
    resp = with_retry(
        lambda: requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Punqle/1.0)"},
            stream=True,
        ),
        exceptions=(requests.RequestException,),
        attempts=2,
    )
    resp.raise_for_status()
    content = resp.raw.read(_MAX_LINK_FETCH_BYTES + 1, decode_content=True)
    if len(content) > _MAX_LINK_FETCH_BYTES:
        raise HTTPException(status_code=400, detail="That page is too large to fetch.")
    return content, resp.headers.get("Content-Type", "")


def _fetch_product_from_link(url: str) -> dict:
    """Free — no AI call, just an HTTP fetch + Open Graph tag parse.
    Works generically across Shopify, WooCommerce, and most storefronts
    without needing a platform-specific integration, since og:title/
    og:description/og:image are the same convention everywhere."""
    html_bytes, _ = _fetch_url_bytes(url)
    soup = BeautifulSoup(html_bytes, "html.parser")

    def meta(*names: str) -> Optional[str]:
        for name in names:
            tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return None

    title = meta("og:title", "twitter:title")
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()
    description = meta("og:description", "twitter:description", "description") or ""
    image_url = meta("og:image", "twitter:image")

    image_base64 = None
    mime_type = None
    if image_url:
        try:
            image_bytes, content_type = _fetch_url_bytes(image_url)
            image_base64 = base64.b64encode(image_bytes).decode("ascii")
            mime_type = content_type or "image/jpeg"
        except (HTTPException, requests.RequestException):
            pass  # title/description alone are still useful without a photo

    return {
        "title": title or "",
        "description": description,
        "image_base64": image_base64,
        "mime_type": mime_type,
    }


_MAX_IMPORTED_PRODUCTS = 200
_MAX_CSV_BYTES = 2 * 1024 * 1024

# Matched case-insensitively against a CSV's header row — covers both our
# own simple template (name/description/price/image_url) and real-world
# Shopify/WooCommerce export headers, so a seller can upload a file
# straight from their existing store without reformatting it first.
_PRODUCT_CSV_COLUMN_ALIASES = {
    "name": {"name", "title", "product name", "product title"},
    "description": {"description", "body (html)", "body", "product description", "details"},
    "price": {"price", "variant price", "cost"},
    "image_url": {"image_url", "image url", "image_src", "image src", "image", "photo", "photo url", "picture"},
}


def _parse_product_csv(content: bytes) -> list[dict]:
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="That CSV file looks empty.")

    column_map: dict[str, str] = {}
    for field, aliases in _PRODUCT_CSV_COLUMN_ALIASES.items():
        for header in reader.fieldnames:
            if header and header.strip().lower() in aliases:
                column_map[field] = header
                break

    if "name" not in column_map:
        raise HTTPException(
            status_code=400,
            detail="Couldn't find a product name column. Make sure your CSV has a 'name' (or 'Title') column.",
        )

    products = []
    for row in reader:
        name = (row.get(column_map["name"]) or "").strip()
        if not name:
            continue
        # Shopify's "Body (HTML)" column is raw HTML — stripping tags here
        # keeps it out of both the catalog list display and the AI prompt
        # this text later feeds into when a product is picked.
        raw_description = (row.get(column_map.get("description", ""), "") or "").strip()
        description = BeautifulSoup(raw_description, "html.parser").get_text(" ", strip=True) if raw_description else ""
        products.append({
            "name": name[:200],
            "description": description[:2000] or None,
            "price": (row.get(column_map.get("price", ""), "") or "").strip()[:50] or None,
            "image_url": (row.get(column_map.get("image_url", ""), "") or "").strip()[:2000] or None,
        })
        if len(products) >= _MAX_IMPORTED_PRODUCTS:
            break

    if not products:
        raise HTTPException(status_code=400, detail="No valid product rows found in that CSV.")
    return products


_MAX_ARTICLE_CHARS = 6000  # plenty for GPT to find several distinct angles without blowing up the prompt


def _extract_article_text(url: str) -> tuple[str, str]:
    """Free — fetch + parse only, no AI call. Reuses the same SSRF-safe
    fetch as fetch-product-link, but pulls paragraph body text instead of
    just Open Graph tags — turning an article into several distinct post
    angles needs actual content, not just a title/description."""
    html_bytes, _ = _fetch_url_bytes(url)
    soup = BeautifulSoup(html_bytes, "html.parser")

    title = None
    tag = soup.find("meta", attrs={"property": "og:title"})
    if tag and tag.get("content"):
        title = tag["content"].strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    paragraphs = soup.find_all("p")
    body_text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    body_text = re.sub(r"\s+", " ", body_text).strip()[:_MAX_ARTICLE_CHARS]

    return title or "", body_text


def _generate_post_ideas_from_article(title: str, body_text: str, category: str) -> list[str]:
    category_guidance = CONTENT_PLAN_CATEGORY_GUIDANCE.get(category, CONTENT_PLAN_CATEGORY_GUIDANCE["other"])
    prompt = f"""You are a social media content strategist for a small business.

{category_guidance}

The business owner found this article/blog post and wants social media post ideas inspired by it:

Title: {title or "(no title found)"}
Content: {body_text}

Suggest 6 distinct social media post ideas this business could write, each taking a different angle inspired by the article above (e.g. a tip from it, a reaction to it, a way to relate it to their own product/service). Don't just summarize the article — turn it into original post ideas for this business, in this business's voice. Keep each idea to one short, concrete line.

Respond with ONLY this JSON format, nothing else:
{{"ideas": ["idea 1", "idea 2"]}}
"""
    response = with_retry(
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You turn a blog or news article into original social media post ideas for a small business."},
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
    ideas = parsed.get("ideas") or []
    return [str(i).strip() for i in ideas if str(i).strip()][:8]


def _generate_competitor_analysis(title: str, body_text: str, category: str) -> dict:
    category_guidance = CONTENT_PLAN_CATEGORY_GUIDANCE.get(category, CONTENT_PLAN_CATEGORY_GUIDANCE["other"])
    prompt = f"""You are a marketing strategist helping a small business understand a competitor.

{category_guidance}

Here is publicly visible information about a competitor:
Name/Title: {title or "(unknown)"}
Content: {body_text}

Based ONLY on the information above — don't invent facts, prices, or claims you can't see there — write:
1. A short 2-3 sentence summary of what this competitor seems to focus on or offer.
2. 5 specific, actionable ways this business could differentiate itself or find a content angle the competitor likely isn't using.

Respond with ONLY this JSON format, nothing else:
{{"summary": "...", "differentiation_ideas": ["idea 1", "idea 2"]}}
"""
    response = with_retry(
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You help small business owners understand a competitor and find ways to stand out, grounded only in what's actually given to you."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
        ),
        exceptions=RETRYABLE_OPENAI_ERRORS,
    )
    ai_text = response.choices[0].message.content.strip()
    m = re.search(r"```(?:json)?\n(.*?)```", ai_text, re.S)
    ai_text_clean = m.group(1).strip() if m else ai_text.strip().strip("`").strip()
    parsed = json.loads(ai_text_clean)
    ideas = parsed.get("differentiation_ideas") or []
    return {
        "summary": str(parsed.get("summary") or "").strip(),
        "differentiation_ideas": [str(i).strip() for i in ideas if str(i).strip()][:6],
    }


# Requesting the shape natively (rather than always generating square and
# cropping/letterboxing afterward client-side) avoids the photo's actual
# subject ever getting cropped into by the story/feed export. The
# client-side export in image-export.ts stays as a fallback path (e.g.
# for images generated before this existed, or if Gemini doesn't follow
# the hint exactly), since compositeImage.ts already adapts to whatever
# dimensions it's actually given.
ASPECT_RATIO_PROMPTS = {
    "square": "Output a high-quality square (1:1) image.",
    "feed": "Output a high-quality image in a 4:5 vertical portrait aspect ratio (taller than it is wide) — a standard Facebook/Instagram feed post shape.",
    "story": "Output a high-quality image in a 9:16 vertical portrait aspect ratio (tall and narrow) — a standard Instagram/Facebook Story shape.",
}

# The prompt-text hint above is not reliable on its own — tested and
# confirmed Gemini ignores it and returns a square image regardless. The
# actual mechanism is this SDK-level config (supported aspect ratios per
# https://ai.google.dev/gemini-api/docs/image-generation: 1:1, 2:3, 3:2,
# 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9).
ASPECT_RATIO_GEMINI_VALUES = {
    "square": "1:1",
    "feed": "4:5",
    "story": "9:16",
}


def _generate_banner_image(image_bytes: bytes, mime_type: str, item_description: str, aspect_ratio: str = "square") -> bytes:
    """Edits the user's own photo (background only) via Gemini —
    deliberately does NOT ask the model to add any text to the image. Text
    rendered by image models is unreliable/garbled; the caller overlays
    real text on this background separately."""
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="AI image generation isn't enabled yet.")

    shape_instruction = ASPECT_RATIO_PROMPTS.get(aspect_ratio, ASPECT_RATIO_PROMPTS["square"])
    prompt = (
        "Edit this product photo into a clean, professional promotional banner "
        "background suitable for a Facebook ad for a small business. "
        "Keep the actual product in the photo exactly as it is — do not change, "
        "replace, warp, or redraw the product itself. Only improve or replace the "
        "background: make it clean, well-lit, and visually appealing, contextually "
        f"fitting for this item/offer: {item_description}. "
        "Do NOT add any text, letters, numbers, or words anywhere in the image — "
        "leave clean, uncluttered space (e.g. near the top or bottom) where text "
        f"will be added afterward by a separate step. {shape_instruction}"
    )
    response = with_retry(
        lambda: gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=genai_types.GenerateContentConfig(
                image_config=genai_types.ImageConfig(
                    aspect_ratio=ASPECT_RATIO_GEMINI_VALUES.get(aspect_ratio, "1:1"),
                ),
            ),
        ),
        exceptions=(Exception,),
        attempts=2,
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    raise Exception("Gemini did not return an image")


def _generate_ai_banner_image(item_description: str, category: str = "other", aspect_ratio: str = "square") -> bytes:
    """Generates a banner image from scratch (no real photo) for users
    without one to upload. The pictured product is AI-imagined rather than
    the business's actual item, so this only ever runs when the frontend
    explicitly sends no file — never a silent fallback for a failed upload."""
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="AI image generation isn't enabled yet.")

    category_guidance = CONTENT_PLAN_CATEGORY_GUIDANCE.get(category, CONTENT_PLAN_CATEGORY_GUIDANCE["other"])
    shape_instruction = ASPECT_RATIO_PROMPTS.get(aspect_ratio, ASPECT_RATIO_PROMPTS["square"])
    prompt = (
        "Generate a clean, professional, photorealistic promotional banner image "
        "for a Facebook ad for a small business. "
        f"Context: {category_guidance} "
        f"The image should visually represent this product/offer: {item_description}. "
        "Make it well-lit, visually appealing, and contextually appropriate. "
        "Do NOT add any text, letters, numbers, or words anywhere in the image — "
        "leave clean, uncluttered space (e.g. near the top or bottom) where text "
        f"will be added afterward by a separate step. {shape_instruction}"
    )
    response = with_retry(
        lambda: gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
            config=genai_types.GenerateContentConfig(
                image_config=genai_types.ImageConfig(
                    aspect_ratio=ASPECT_RATIO_GEMINI_VALUES.get(aspect_ratio, "1:1"),
                ),
            ),
        ),
        exceptions=(Exception,),
        attempts=2,
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    raise Exception("Gemini did not return an image")


async def _get_banner_image(
    image_bytes: Optional[bytes],
    mime_type: Optional[str],
    item_description: str,
    category: str,
    aspect_ratio: str = "square",
) -> bytes:
    # Gemini image generation is a blocking call and can take well over a
    # minute (especially generating from scratch, no reference photo) — run
    # it off the event loop so a slow generation doesn't stall every other
    # request this worker is handling, health checks included. (This fixes
    # a real bug: without run_in_threadpool, one slow generation freezes
    # the whole backend for every user until it finishes.)
    if image_bytes:
        return await run_in_threadpool(_generate_banner_image, image_bytes, mime_type or "image/jpeg", item_description, aspect_ratio)
    return await run_in_threadpool(_generate_ai_banner_image, item_description, category, aspect_ratio)


def _remove_background(image_bytes: bytes, mime_type: str) -> bytes:
    """Swaps the photo's background for clean flat white, keeping the
    product itself untouched — same edit-not-generate pattern as
    _generate_banner_image, just a narrower prompt."""
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="AI image generation isn't enabled yet.")

    prompt = (
        "Edit this photo: remove the background completely and replace it with "
        "a clean, plain, solid white background. Keep the actual product/subject "
        "in the photo exactly as it is — do not change, replace, warp, or redraw "
        "it. Do NOT add any text, letters, numbers, or words anywhere in the "
        "image. Output a high-quality image with only the background changed."
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


def _enhance_image(image_bytes: bytes, mime_type: str) -> bytes:
    """Improves lighting, sharpness, and color on the user's own photo
    without altering the product/composition — a cleanup pass, not a
    background change."""
    if gemini_client is None:
        raise HTTPException(status_code=503, detail="AI image generation isn't enabled yet.")

    prompt = (
        "Edit this photo to improve its quality: fix lighting, increase "
        "sharpness, correct color balance, and reduce noise/blur so it looks "
        "professionally shot. Keep the actual product/subject, composition, "
        "and background exactly as they are — only improve technical image "
        "quality, do not add, remove, or move anything in the scene. Do NOT "
        "add any text, letters, numbers, or words anywhere in the image."
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


@app.post("/ads/generate", response_model=AdGenerateResponse, tags=["ads"])
@limiter.limit("10/minute")
async def generate_ad(
    request: Request,
    item_description: str,
    aspect_ratio: str = "square",
    file: Optional[UploadFile] = File(None),
    user_id: str = Depends(get_current_user_id),
):
    try:
        if aspect_ratio not in ASPECT_RATIO_PROMPTS:
            aspect_ratio = "square"
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
        banner_bytes = await _get_banner_image(image_bytes, mime_type, item_description, category, aspect_ratio)

        new_credits = _spend_ad_credit(user_id)
        banner_b64 = base64.b64encode(banner_bytes).decode("ascii")
        _save_generated_post(user_id, item_description, copy[0], banner_b64)

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


@app.post("/ads/generate-image-variant", response_model=AdImageVariantResponse, tags=["ads"])
@limiter.limit("10/minute")
async def generate_ad_image_variant(
    request: Request,
    item_description: str,
    aspect_ratio: str = "square",
    file: Optional[UploadFile] = File(None),
    user_id: str = Depends(get_current_user_id),
):
    """A second (or third) banner background option for the same post —
    unlike the bundled-free caption variants, each extra image costs its
    own credit since Gemini image generation is the expensive part of a
    generation, not the text."""
    try:
        if aspect_ratio not in ASPECT_RATIO_PROMPTS:
            aspect_ratio = "square"
        credits = _get_ad_credits(user_id)
        if credits <= 0:
            raise HTTPException(
                status_code=402,
                detail="You're out of ad credits. Upgrade to keep generating.",
            )

        image_bytes = await file.read() if file is not None else None
        mime_type = file.content_type if file is not None else None
        category = _get_business_category(user_id)
        banner_bytes = await _get_banner_image(image_bytes, mime_type, item_description, category, aspect_ratio)

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


VIDEO_CREDIT_COST = 10  # ~10x an image credit, matching Veo 3.1 Lite's real ~$0.40/8s-720p vs an image's ~$0.04
VEO_MODEL = "veo-3.1-lite-generate-preview"
VIDEO_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "VideoOverlay-Bold.ttf")


def _generate_video_headline(item_description: str, category: str) -> str:
    """Free — a short on-screen hook, distinct from the longer Facebook
    caption text (_generate_ad_copy): Veo has no reliable way to render
    text itself (same limitation as the image model), so a real headline
    only exists once burned on afterward — this generates what to burn.
    Kept short on purpose so it fits one line at a legible size without
    needing the wrapping logic compositeImage.ts uses for images."""
    category_guidance = CONTENT_PLAN_CATEGORY_GUIDANCE.get(category, CONTENT_PLAN_CATEGORY_GUIDANCE["other"])
    prompt = f"""You are writing a short on-screen text hook for a social media ad video for a small business.

{category_guidance}

The video is about: {item_description}

Write ONE short, punchy headline for this video — 4 to 8 words, under 40 characters, no hashtags, no emoji, no quotation marks. It should work as a bold on-screen caption overlaid on the video, not a full sentence.

Respond with ONLY the headline text, nothing else.
"""
    response = with_retry(
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write short, punchy on-screen video hooks for small business ads."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        ),
        exceptions=RETRYABLE_OPENAI_ERRORS,
    )
    headline = response.choices[0].message.content.strip().strip('"').strip("'")
    return headline[:60]


def _escape_ffmpeg_drawtext(text: str) -> str:
    """drawtext's own mini-syntax treats \\, ', %, and : as special —
    escape in the order that keeps escaping itself from being re-escaped."""
    return (
        text.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace(":", "\\:")
        .replace("'", "\\'")
    )


def _burn_text_on_video(video_bytes: bytes, headline: str) -> bytes:
    """Bakes the headline onto the video as a bottom caption bar, the
    same visual role compositeImage.ts's caption bar plays for images —
    real text the AI model itself can't reliably render. ffmpeg comes
    from imageio-ffmpeg's bundled binary, no system ffmpeg install
    needed (same reasoning as this app's earlier frame-extraction work)."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    escaped = _escape_ffmpeg_drawtext(headline)
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "input.mp4")
        output_path = os.path.join(tmp_dir, "output.mp4")
        with open(input_path, "wb") as f:
            f.write(video_bytes)

        # Sized relative to the video's own width/height (not fixed
        # pixels) — a fixed 44px font tuned for a 1280px-wide 16:9 frame
        # overflowed both edges on a 720px-wide 9:16 frame once the
        # aspect-ratio picker shipped. Ratios below match what the old
        # fixed values worked out to on the original 1280x720 frame.
        vf = (
            f"drawtext=fontfile={VIDEO_FONT_PATH}:text='{escaped}':"
            "fontcolor=white:fontsize=w*0.0344:box=1:boxcolor=black@0.6:boxborderw=w*0.01875:"
            "x=(w-text_w)/2:y=h-text_h-h*0.0694"
        )
        cmd = [ffmpeg_exe, "-y", "-i", input_path, "-vf", vf, "-c:a", "copy", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("ffmpeg drawtext failed: %s", result.stderr[-2000:])
            raise Exception("Couldn't add text to the video.")

        with open(output_path, "rb") as f:
            return f.read()


@app.post("/ads/generate-video", response_model=VideoOperationOut, tags=["ads"])
@limiter.limit("5/minute")
def start_video_generation(
    request: Request,
    req: GenerateVideoRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Starts a Veo video generation job. This alone doesn't spend a
    credit — video generation genuinely takes 1-2+ minutes, far past any
    reasonable HTTP request timeout, so this only kicks the job off and
    hands back an opaque operation handle for the frontend to poll via
    /ads/video-status. Credits are only spent there, once generation
    actually succeeds, same as every other AI action in this app."""
    try:
        if gemini_client is None:
            raise HTTPException(status_code=503, detail="Video generation isn't available right now.")
        credits = _get_ad_credits(user_id)
        if credits < VIDEO_CREDIT_COST:
            raise HTTPException(
                status_code=402,
                detail=f"Video needs {VIDEO_CREDIT_COST} credits — you have {credits}.",
            )
        item_description = (req.item_description or "").strip()
        if not item_description:
            raise HTTPException(status_code=400, detail="Tell us what the video is about.")

        image = None
        if req.image_base64:
            image = genai_types.Image(
                image_bytes=base64.b64decode(req.image_base64),
                mime_type=req.image_mime_type or "image/jpeg",
            )

        category = _get_business_category(user_id)
        headline = _generate_video_headline(item_description, category)

        prompt = (
            f"A short, eye-catching social media ad video for a small business. {item_description}. "
            "Professional, well-lit, realistic. No on-screen text, captions, or logos."
        )
        operation = gemini_client.models.generate_videos(
            model=VEO_MODEL,
            prompt=prompt,
            image=image,
            config=genai_types.GenerateVideosConfig(
                aspect_ratio=req.aspect_ratio,
                resolution="720p",
                duration_seconds="8",
            ),
        )
        return {"operation": operation.model_dump(mode="json"), "headline": headline}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ads/video-status", response_model=VideoStatusResponse, tags=["ads"])
@limiter.limit("30/minute")
def check_video_status(
    request: Request,
    req: VideoStatusRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Stateless polling — the operation handle round-trips through the
    client instead of being kept in server memory, so a mid-generation
    backend restart/redeploy on Render doesn't strand anyone's job."""
    try:
        if gemini_client is None:
            raise HTTPException(status_code=503, detail="Video generation isn't available right now.")
        operation = genai_types.GenerateVideosOperation.model_validate(req.operation)
        operation = gemini_client.operations.get(operation)
        if not operation.done:
            return {"done": False, "video_base64": None, "credits_remaining": None}

        if operation.error:
            raise HTTPException(status_code=502, detail="Video generation failed. Please try again.")

        result = operation.result or operation.response
        if not result or not result.generated_videos:
            raise HTTPException(status_code=502, detail="Video generation didn't return a video. Please try again.")
        generated = result.generated_videos[0]
        video_bytes = gemini_client.files.download(file=generated.video)

        headline = (req.headline or "").strip()
        if headline:
            try:
                video_bytes = _burn_text_on_video(video_bytes, headline)
            except Exception as e:
                # The raw video is still a perfectly usable result without
                # its caption — not worth failing (and losing) an already-
                # generated, already-about-to-be-charged-for video over a
                # text-overlay step failing.
                logger.error("Text burn-in failed, returning video without it: %s", str(e), exc_info=True)

        new_credits = _spend_ad_credits(user_id, VIDEO_CREDIT_COST)
        return {
            "done": True,
            "video_base64": base64.b64encode(video_bytes).decode("ascii"),
            "credits_remaining": new_credits,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ads/remove-background", response_model=AdImageVariantResponse, tags=["ads"])
@limiter.limit("10/minute")
async def remove_background(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """Standalone quick-edit tool — same 1-credit-per-image-call pricing
    as the main generation, since this hits the same paid Gemini image
    model."""
    try:
        credits = _get_ad_credits(user_id)
        if credits <= 0:
            raise HTTPException(
                status_code=402,
                detail="You're out of ad credits. Upgrade to keep generating.",
            )

        image_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        result_bytes = await run_in_threadpool(_remove_background, image_bytes, mime_type)

        new_credits = _spend_ad_credit(user_id)

        return {
            "banner_image_base64": base64.b64encode(result_bytes).decode("ascii"),
            "credits_remaining": new_credits,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ads/enhance-image", response_model=AdImageVariantResponse, tags=["ads"])
@limiter.limit("10/minute")
async def enhance_image(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """Standalone quick-edit tool — same 1-credit-per-image-call pricing
    as the main generation, since this hits the same paid Gemini image
    model."""
    try:
        credits = _get_ad_credits(user_id)
        if credits <= 0:
            raise HTTPException(
                status_code=402,
                detail="You're out of ad credits. Upgrade to keep generating.",
            )

        image_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        result_bytes = await run_in_threadpool(_enhance_image, image_bytes, mime_type)

        new_credits = _spend_ad_credit(user_id)

        return {
            "banner_image_base64": base64.b64encode(result_bytes).decode("ascii"),
            "credits_remaining": new_credits,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ads/translate-captions", response_model=TranslateCaptionsResponse, tags=["ads"])
@limiter.limit("15/minute")
def translate_captions(
    request: Request,
    req: TranslateCaptionsRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Free — text-only, translates captions already paid for by the
    original generation. No credit spend."""
    try:
        target_language = (req.target_language or "").strip()
        if not target_language:
            raise HTTPException(status_code=400, detail="Pick a language to translate into.")
        if not req.captions:
            raise HTTPException(status_code=400, detail="No captions to translate.")

        captions_in = [c.model_dump() for c in req.captions]
        translated = _translate_captions(captions_in, target_language)
        if not translated:
            raise HTTPException(status_code=502, detail="Translation failed. Please try again.")

        return {"captions": translated}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ads/suggest-hashtags", response_model=SuggestHashtagsResponse, tags=["ads"])
@limiter.limit("15/minute")
def suggest_hashtags(
    request: Request,
    req: SuggestHashtagsRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Free — text-only GPT call, same economics as translate-captions."""
    try:
        text = (req.item_description or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Tell us what the post is about first.")
        category = _get_business_category(user_id)
        hashtags = _suggest_hashtags(text, category)
        return {"hashtags": hashtags}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ads/idea-labs", response_model=IdeaLabsResponse, tags=["ads"])
@limiter.limit("10/minute")
def idea_labs(request: Request, user_id: str = Depends(get_current_user_id)):
    """Free — text-only GPT call. General post inspiration for a user
    with no specific product typed in yet, grounded only in their saved
    business category."""
    try:
        category = _get_business_category(user_id)
        ideas = _generate_idea_labs_ideas(category)
        return {"ideas": ideas}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ads/fetch-product-link", response_model=FetchProductLinkResponse, tags=["ads"])
@limiter.limit("10/minute")
def fetch_product_link(
    request: Request,
    req: FetchProductLinkRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Free — a plain HTTP fetch + HTML parse, no AI call. Lets an
    e-commerce seller paste their product page link instead of typing a
    description and uploading a photo from scratch."""
    try:
        url = (req.url or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="Paste a product page link.")
        result = _fetch_product_from_link(url)
        if not result["title"] and not result["description"]:
            raise HTTPException(status_code=422, detail="Couldn't find product info at that link.")
        return result
    except HTTPException:
        raise
    except requests.RequestException:
        raise HTTPException(status_code=400, detail="Couldn't reach that link.")
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/products/import-csv", response_model=ImportedProductsListResponse, tags=["products"])
@limiter.limit("5/minute")
async def import_products_csv(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """Replaces the user's whole saved catalog with what's in this file —
    "here's my current product list," not an accumulating import log, so
    re-uploading a corrected file doesn't leave stale duplicates behind."""
    try:
        content = await file.read()
        if len(content) > _MAX_CSV_BYTES:
            raise HTTPException(status_code=400, detail="That file is too large (max 2MB).")
        products = _parse_product_csv(content)
        with_retry(lambda: supabase.table("imported_products").delete().eq("owner_id", user_id).execute())
        rows = [{"owner_id": user_id, **p} for p in products]
        res = with_retry(lambda: supabase.table("imported_products").insert(rows).execute())
        res = ensure_supabase_response(res, "import products")
        return {"products": res.data}
    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Couldn't read that file — please upload a CSV.")
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products", response_model=ImportedProductsListResponse, tags=["products"])
def list_products(user_id: str = Depends(get_current_user_id)):
    try:
        res = with_retry(lambda: supabase.table("imported_products")
            .select("id, name, description, price, image_url")
            .eq("owner_id", user_id)
            .order("created_at", desc=False)
            .execute())
        res = ensure_supabase_response(res, "list products")
        return {"products": res.data}
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/products", tags=["products"])
def clear_products(user_id: str = Depends(get_current_user_id)):
    try:
        supabase.table("imported_products").delete().eq("owner_id", user_id).execute()
        return {"deleted": True}
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/products/fetch-image", response_model=FetchProductImageResponse, tags=["products"])
@limiter.limit("20/minute")
def fetch_product_image(request: Request, req: FetchProductImageRequest, user_id: str = Depends(get_current_user_id)):
    """Same SSRF-safe fetch as fetch-stock-photo, but not restricted to a
    single host — CSV-imported image URLs can point at any store's own
    CDN (Shopify, WooCommerce, etc.), unlike the curated Pexels search."""
    try:
        url = (req.url or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="No image URL provided.")
        image_bytes, content_type = _fetch_url_bytes(url)
        return {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "mime_type": content_type or "image/jpeg",
        }
    except HTTPException:
        raise
    except requests.RequestException:
        raise HTTPException(status_code=400, detail="Couldn't fetch that image.")
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ads/blog-to-posts", response_model=BlogToPostsResponse, tags=["ads"])
@limiter.limit("8/minute")
def blog_to_posts(
    request: Request,
    req: BlogToPostsRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Free — one fetch + one text-only GPT call. Turns a blog/article
    link into several distinct post ideas, unlike fetch-product-link
    which extracts one product's info — an article has no single product
    to pull out, so this generates ideas inspired by it instead."""
    try:
        url = (req.url or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="Paste a blog or article link.")
        title, body_text = _extract_article_text(url)
        if not body_text:
            raise HTTPException(status_code=422, detail="Couldn't find any article content at that link.")
        category = _get_business_category(user_id)
        ideas = _generate_post_ideas_from_article(title, body_text, category)
        if not ideas:
            raise HTTPException(status_code=502, detail="Couldn't come up with ideas from that article. Try another link.")
        return {"title": title, "ideas": ideas}
    except HTTPException:
        raise
    except requests.RequestException:
        raise HTTPException(status_code=400, detail="Couldn't reach that link.")
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ads/competitor-analysis", response_model=CompetitorAnalysisResponse, tags=["ads"])
@limiter.limit("8/minute")
def competitor_analysis(
    request: Request,
    req: CompetitorAnalysisRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Free — one fetch + one text-only GPT call, same pattern as
    blog-to-posts. Works best on a competitor's own website; Facebook/
    Instagram pages are JS-rendered, so only their public link-preview
    meta tags (title/description) are reliably visible to a plain HTTP
    fetch, not their actual post feed — no scraping login-gated content."""
    try:
        url = (req.url or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="Paste a competitor's website or page link.")
        title, body_text = _extract_article_text(url)
        if not title and not body_text:
            raise HTTPException(status_code=422, detail="Couldn't find any content at that link.")
        category = _get_business_category(user_id)
        result = _generate_competitor_analysis(title, body_text, category)
        if not result["summary"]:
            raise HTTPException(status_code=502, detail="Couldn't analyze that link. Try another one.")
        return {"competitor_name": title or "Competitor", **result}
    except HTTPException:
        raise
    except requests.RequestException:
        raise HTTPException(status_code=400, detail="Couldn't reach that link.")
    except Exception as e:
        logger.error("ERROR: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ads/stock-photos", response_model=StockPhotoSearchResponse, tags=["ads"])
@limiter.limit("20/minute")
def search_stock_photos(request: Request, query: str, user_id: str = Depends(get_current_user_id)):
    """Free — a search proxy to Pexels, no AI call. A photo source
    option for users with no product photo of their own and who don't
    want an AI-imagined one."""
    if not PEXELS_API_KEY:
        raise HTTPException(status_code=503, detail="Stock photo search isn't enabled yet.")
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Type something to search for.")
    try:
        resp = with_retry(
            lambda: requests.get(
                "https://api.pexels.com/v1/search",
                params={"query": q, "per_page": 15, "orientation": "square"},
                headers={"Authorization": PEXELS_API_KEY},
                timeout=10,
            ),
            exceptions=(requests.RequestException,),
            attempts=2,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Couldn't reach the stock photo service.")
    results = [
        {
            "id": str(p["id"]),
            "thumbnail_url": p["src"]["medium"],
            "full_url": p["src"]["large"],
            "photographer": p.get("photographer", ""),
        }
        for p in data.get("photos", [])
    ]
    return {"results": results}


@app.post("/ads/fetch-stock-photo", response_model=FetchStockPhotoResponse, tags=["ads"])
@limiter.limit("20/minute")
def fetch_stock_photo(request: Request, req: FetchStockPhotoRequest, user_id: str = Depends(get_current_user_id)):
    """Free — fetches the actual image bytes for a photo the user picked
    from search results. Restricted to Pexels' own CDN host (not an
    arbitrary user-supplied URL like fetch-product-link) as defense in
    depth on top of the existing SSRF guard in _fetch_url_bytes."""
    url = (req.url or "").strip()
    if urlparse(url).hostname != "images.pexels.com":
        raise HTTPException(status_code=400, detail="That link isn't allowed.")
    try:
        image_bytes, content_type = _fetch_url_bytes(url)
    except HTTPException:
        raise
    except requests.RequestException:
        raise HTTPException(status_code=400, detail="Couldn't fetch that photo.")
    return {
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "mime_type": content_type or "image/jpeg",
    }


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
