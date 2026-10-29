import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from acl_anthology import Anthology
from git import Repo
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tqdm import tqdm

from anthology_rag.config import EMBEDDING_BATCH_SIZE as BATCH_SIZE
from anthology_rag.config import EMBEDDING_MODEL


def get_embeddings() -> Embeddings:
    # Imported lazily so tests that use a fake embedding don't load torch.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True}
    )


def get_store(persist_dir: Path, embeddings: Embeddings) -> Chroma:
    # Normalized embeddings + cosine distance (set here, fixed at collection creation).
    return Chroma(
        collection_name="papers",
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_anthology(repo_path: Path | None = None) -> tuple[Anthology, str]:
    """Clone or pull the anthology repo; returns it with the current commit hash."""
    anthology = Anthology.from_repo(path=repo_path)
    version = Repo(anthology.datadir.parent).head.commit.hexsha
    return anthology, version


def paper_documents(anthology: Anthology) -> Iterator[Document]:
    """One Document per paper; the anthology ID (e.g. 2023.acl-long.1) is the Document id."""
    for paper in anthology.papers():
        # Skip volume front matter and withdrawn papers; they are not real papers.
        if paper.is_deleted or paper.is_frontmatter or paper.title is None:
            continue
        title = paper.title.as_text()
        abstract = paper.abstract.as_text() if paper.abstract else ""
        # This text is what gets embedded; many older papers have no abstract.
        text = f"{title}\n{abstract}".strip()
        venues = list(paper.venue_ids)
        metadata: dict[str, str | int | bool] = {
            "title": title,
            "abstract": abstract,
            # Chroma metadata must be scalar, so lists are stored as joined strings.
            "authors": "; ".join(a.name.as_full() for a in paper.authors),
            "year": int(paper.year),
            "venues": ",".join(venues),
            "url": paper.web_url or "",
            # Lets updates detect changed papers without re-embedding everything.
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        }
        # Chroma has no "list contains" filter, so each venue gets its own flag.
        metadata.update({f"venue_{v}": True for v in venues})
        yield Document(id=paper.full_id, page_content=text, metadata=metadata)


def build_or_update(
    anthology: Anthology,
    version: str,
    persist_dir: Path,
    embeddings: Embeddings,
    max_papers: int | None = None,
) -> Chroma:
    """Create the vector store, or bring an existing one up to date.

    Papers are keyed by anthology ID: new ones are added, removed ones deleted and
    ones whose title/abstract changed are re-embedded. Nothing happens if the
    anthology version and paper limit are the ones already indexed. With max_papers,
    only that many papers with the latest publication years are indexed.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    # meta.json remembers which anthology commit and paper limit the store was last synced to.
    meta_file = persist_dir / "meta.json"
    store = get_store(persist_dir, embeddings)
    meta = {"version": version, "max_papers": max_papers}
    if meta_file.exists() and json.loads(meta_file.read_text()) == meta:
        return store

    # Diff the anthology (docs) against what is already stored (old_hashes), by paper ID.
    all_docs = [d for d in paper_documents(anthology) if d.id]
    if max_papers is not None:
        # Stable sort: papers of the same year keep anthology order, so the cut is deterministic.
        all_docs = sorted(all_docs, key=lambda d: d.metadata["year"], reverse=True)[:max_papers]
    docs = {d.id: d for d in all_docs}
    indexed = store.get(include=["metadatas"])
    old_hashes = {i: m["content_hash"] for i, m in zip(indexed["ids"], indexed["metadatas"])}

    removed = [i for i in old_hashes if i not in docs]
    # "Changed" covers both new papers (no old hash) and edited ones (different hash).
    changed = [d for i, d in docs.items() if old_hashes.get(i) != d.metadata["content_hash"]]
    # Batched so memory stays bounded and a crash loses little (meta.json is written last,
    # so an interrupted build is simply resumed by the next run).
    if removed or changed:
        with tqdm(
            total=len(removed) + len(changed), desc="Updating database", unit="paper"
        ) as progress:
            if removed:
                store.delete(ids=removed)
                progress.update(len(removed))
            for start in range(0, len(changed), BATCH_SIZE):
                batch = changed[start : start + BATCH_SIZE]
                # Replace rather than upsert so stale metadata keys (e.g. old venues) vanish.
                stale = [d.id for d in batch if d.id in old_hashes]
                if stale:
                    store.delete(ids=stale)
                store.add_documents(batch, ids=[d.id for d in batch])
                progress.update(len(batch))

    meta_file.write_text(json.dumps(meta))
    return store
