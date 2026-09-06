# Nexamind.Ai

**Think. Search. Understand.**

Nexamind.Ai is a general-purpose AI assistant built on the existing Multi-Agent RAG research code. It can answer stable questions with an LLM, search the web for current information, use the existing HotpotQA/RAG knowledge base, or combine web + RAG evidence.

## Features

- General LLM answers through OpenRouter/OpenAI-compatible APIs
- Deterministic question routing: `LLM_ONLY`, `WEB_SEARCH`, `RAG`, `HYBRID`
- Zero-key DuckDuckGo HTML search for local development
- Optional Google Programmable Search JSON API provider
- Bounded, public-destination web fetching with SSRF protections
- Source cards and grounded web evidence
- Bounded in-process conversation memory and follow-up context
- Modern responsive chat UI with dark mode
- Existing research endpoint `/query` preserved

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn src.api.app:app --reload
```

Open `http://127.0.0.1:8000/`.

## OpenRouter

Set `OPENAI_API_KEY`, `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, `LLM_PROVIDER=openai`, and your chosen `LLM_MODEL` in `.env`. Never commit `.env` or expose the key to the browser.

## Web search

The default `WEB_SEARCH_PROVIDER=duckduckgo` needs no search API key and is intended for development. To use Google Programmable Search, set `WEB_SEARCH_PROVIDER=google`, `GOOGLE_API_KEY`, and `GOOGLE_CSE_ID`.

## API

`POST /api/chat` accepts `message`, optional `conversation_id`, `web_enabled`, and `rag_enabled`.

The original research API remains available at `POST /query`, alongside `/health` and `/examples`.

## Research layer

The project preserves the existing retrieval, agents, orchestration, HotpotQA, evaluation, experiments, and tests. Nexamind.Ai is the user-facing layer built around that research foundation; it does not replace the research implementation.

## Testing

```powershell
python -m pytest -m "not slow"
```

Some real-data tests require `pyarrow` and the HotpotQA files; the standard requirements include `pyarrow`.
