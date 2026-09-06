# Nexamind.Ai

Nexamind.Ai is the user-facing assistant layer built on the existing Multi-Agent RAG research code. It routes questions to a general LLM, web search, the existing knowledge base, or a hybrid of web + RAG.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.api.app:app --reload
```

Open `http://127.0.0.1:8000/`.

## Configuration

Copy `.env.example` to `.env` and configure your OpenRouter credentials. Never commit `.env`.

For local development, the default web search provider is DuckDuckGo HTML search and does not require a search API key. The provider is isolated behind `src/web/search.py` so it can be replaced later.

## Routes

- `LLM_ONLY` — stable/general questions
- `WEB_SEARCH` — current, recent and news questions
- `RAG` — questions explicitly referring to the knowledge base
- `HYBRID` — questions needing both web and knowledge-base evidence

## API

`POST /api/chat` accepts `message`, optional `conversation_id`, `web_enabled`, and `rag_enabled`.

The original research endpoint `POST /query` remains available for experiments and compatibility.

## Safety and reliability

The web fetcher uses HTTPS/HTTP validation, public-destination checks, timeouts, bounded content extraction, and no frontend access to provider secrets. User-facing errors do not expose stack traces.

Conversation memory is intentionally in-process and bounded for local development. A production deployment should replace it with a persistent store and add authentication/rate limiting.
