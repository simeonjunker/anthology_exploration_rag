# ACL Anthology RAG

Chatbot that finds ACL Anthology papers from natural-language requests. It keeps multi-turn
history, so follow-ups like "only from 2023" work.

- `anthology_rag.db`: standalone database component (acl-anthology API → Chroma, embeddings
  `BAAI/bge-small-en-v1.5` over title + abstract; authors, year and venues are metadata).
- `anthology_rag.agent`: LangGraph flow `parse_query → retrieve → respond`.

## Usage

```sh
uv sync
uv run build-db     # first run: full build; later runs: pull the anthology and update
uv run chat         # needs an OpenAI-compatible server (llama.cpp / vLLM) on 127.0.0.1:8345
```

`build-db` is incremental: it records the anthology commit, skips if unchanged, otherwise
adds new papers, deletes removed ones and re-embeds only those whose title/abstract changed.

Chat commands: `/reset` starts a new conversation, `/quit` exits.

## Configuration

Models, server address, paths and main hyperparameters come from environment variables,
optionally set in a `.env` file (`cp .env.example .env`). All have defaults; see
[.env.example](.env.example) and `src/anthology_rag/config.py`.

## Development

```sh
uv run pytest      # fake LLM and embeddings, no network
uv run ruff check . && uv run ruff format --check .
```
