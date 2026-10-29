import hashlib
from types import SimpleNamespace

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing, so shared words mean similarity."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * 128
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1.0
        return vec


def make_paper(pid, title, abstract, authors, year, venues):
    as_text = lambda s: SimpleNamespace(as_text=lambda: s)  # noqa: E731
    return SimpleNamespace(
        full_id=pid,
        title=as_text(title),
        abstract=as_text(abstract) if abstract else None,
        authors=[SimpleNamespace(name=SimpleNamespace(as_full=lambda a=a: a)) for a in authors],
        year=str(year),
        venue_ids=venues,
        web_url=f"https://aclanthology.org/{pid}/",
        is_deleted=False,
        is_frontmatter=False,
    )


class FakeAnthology:
    def __init__(self, papers):
        self._papers = papers

    def papers(self):
        return iter(self._papers)


PAPERS = [
    make_paper(
        "2020.acl-1",
        "Neural machine translation",
        "We translate text.",
        ["Ann Lee"],
        2020,
        ["acl"],
    ),
    make_paper(
        "2022.emnlp-1",
        "Neural machine translation again",
        "Better translation.",
        ["Bob Ray"],
        2022,
        ["emnlp"],
    ),
    make_paper(
        "2023.acl-1",
        "Neural machine translation at scale",
        "Huge translation models.",
        ["Ann Lee"],
        2023,
        ["acl", "ws"],
    ),
    make_paper(
        "2023.acl-2",
        "Speech recognition with transformers",
        "Audio models.",
        ["Cy Dee"],
        2023,
        ["acl"],
    ),
]
