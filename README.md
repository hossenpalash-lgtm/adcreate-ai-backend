# AdCreate.AI Backend

FastAPI backend for AdCreate.AI — AI-generated Facebook ad copy, banner images, and weekly content plans for small businesses.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
uvicorn main:app --reload
```

## Environment variables

See `.env.example`. `OPENAI_API_KEY`, `SUPABASE_URL`, and `SUPABASE_KEY` are required at startup. `GEMINI_API_KEY` is optional — without it, AI banner-image generation returns a 503 but the rest of the app keeps working.

## Deploy

Deployed as a Render web service:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Enable auto-deploy on push to `main`
