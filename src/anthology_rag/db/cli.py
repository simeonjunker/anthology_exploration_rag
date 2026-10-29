from anthology_rag.config import ANTHOLOGY_REPO_PATH, MAX_PAPERS, PERSIST_DIR
from anthology_rag.db import build_or_update, get_embeddings, load_anthology


def main() -> None:
    anthology, version = load_anthology(ANTHOLOGY_REPO_PATH)
    store = build_or_update(anthology, version, PERSIST_DIR, get_embeddings(), MAX_PAPERS)
    print(f"Anthology {version[:10]}: {store._collection.count()} papers indexed")
