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

## What you can ask

Each answer starts with a `[search]` line showing the query the model derived, followed by a
numbered list. Every hit shows title, authors, year, venue, a similarity score (cosine, 1.0 =
identical) and the anthology link. The list is built by the code, not the LLM, so scores and
links are exact.

| Interaction | Example |
| --- | --- |
| Topic search | `Papers about <topic>` |
| Filter by year, venue, author | `... from <year> or later`, `... at <venue>`, `... by <author>` |
| Number of results | `<N> papers about <topic>` |
| More results | `more`, `<N> more` (continues the numbering) |
| Refine the last search | `from <year> or later`, `only at <venue>` (restarts the list at 1) |
| Similar papers | `similar papers like <N>` (embeds that paper's title and abstract) |
| Small talk | `thanks`, `hello` get a short reply; the last search stays available |

Notes:

- Without a number you get all papers above `MIN_SIMILARITY`, but at least `MIN_RESULTS` and at
  most `MAX_RESULTS` (see [.env.example](.env.example)). `MAX_RESULTS` is also the page size for
  "more".
- Paper numbers refer to the most recent list, including papers added by "more".
- Years, venues, authors and paper numbers are only used if you actually wrote them; the model
  cannot add them on its own.
- Venues are anthology venue IDs such as `acl`, `emnlp`, `naacl`, `coling`, `lrec`.
- Only the last `MAX_HISTORY_MESSAGES` messages are sent to the model.

## Docker

Build the image:

```sh
docker build -t anthology-rag .
```

Build the database once, storing it and the downloaded embedding model in named volumes:

```sh
docker run --rm \
  -v anthology-data:/data \
  -v hf-cache:/home/app/.cache/huggingface \
  anthology-rag build-db
```

Run the interactive chat against a model server on the host:

```sh
docker run --rm -it \
  --add-host=host.docker.internal:host-gateway \
  -e LLM_BASE_URL=http://host.docker.internal:8345/v1 \
  -v anthology-data:/data \
  -v hf-cache:/home/app/.cache/huggingface \
  anthology-rag
```

On macOS and Windows, omit `--add-host`; Docker provides `host.docker.internal` there. On Linux,
keep it so the container can resolve the host. The model server must accept connections from
Docker, not only from `127.0.0.1`. To use a `.env` file, add `--env-file .env` and keep
`PERSIST_DIR=/data/chroma_db` so the database remains in the mounted volume. Pass additional
configuration with `-e NAME=value` as needed.

## Configuration

Models, server address, paths and main hyperparameters come from environment variables,
optionally set in a `.env` file (`cp .env.example .env`). All have defaults; see
[.env.example](.env.example) and `src/anthology_rag/config.py`.

## Development

```sh
uv run pytest      # fake LLM and embeddings, no network
uv run ruff check . && uv run ruff format --check .
```
