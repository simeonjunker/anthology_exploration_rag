from contextlib import contextmanager
from copy import deepcopy

from anthology_rag.db import build_or_update
from tests.fakes import PAPERS, FakeAnthology, FakeEmbeddings, make_paper


class CountingEmbeddings(FakeEmbeddings):
    def __init__(self):
        self.embedded: list[str] = []

    def embed_documents(self, texts):
        self.embedded += texts
        return super().embed_documents(texts)


def ids(store):
    return set(store.get()["ids"])


def test_build_stores_metadata(tmp_path):
    store = build_or_update(FakeAnthology(PAPERS), "v1", tmp_path, FakeEmbeddings())
    got = store.get(ids=["2023.acl-1"])["metadatas"][0]
    assert got["authors"] == "Ann Lee"
    assert got["year"] == 2023
    assert got["venues"] == "acl,ws"
    assert got["venue_ws"] is True
    assert got["url"] == "https://aclanthology.org/2023.acl-1/"
    assert "Huge translation models." in store.get(ids=["2023.acl-1"])["documents"][0]


def test_same_version_is_skipped(tmp_path):
    emb = CountingEmbeddings()
    build_or_update(FakeAnthology(PAPERS), "v1", tmp_path, emb)
    emb.embedded.clear()
    build_or_update(FakeAnthology(PAPERS), "v1", tmp_path, emb)
    assert emb.embedded == []


def test_update_adds_removes_and_reembeds_only_changes(tmp_path):
    emb = CountingEmbeddings()
    build_or_update(FakeAnthology(PAPERS), "v1", tmp_path, emb)
    emb.embedded.clear()

    papers = deepcopy(PAPERS)[1:]  # 2020.acl-1 removed
    papers[0] = make_paper(
        "2022.emnlp-1",
        "Neural machine translation again",
        "Revised abstract.",
        ["Bob Ray"],
        2022,
        ["emnlp"],
    )
    papers.append(make_paper("2024.acl-1", "Brand new paper", "Fresh.", ["Eve Fox"], 2024, ["acl"]))
    store = build_or_update(FakeAnthology(papers), "v2", tmp_path, emb)

    assert ids(store) == {"2022.emnlp-1", "2023.acl-1", "2023.acl-2", "2024.acl-1"}
    assert sorted(emb.embedded) == sorted(
        [
            "Neural machine translation again\nRevised abstract.",
            "Brand new paper\nFresh.",
        ]
    )
    assert "Revised abstract." in store.get(ids=["2022.emnlp-1"])["documents"][0]


def test_update_reports_progress_for_removed_and_changed_papers(tmp_path, monkeypatch):
    build_or_update(FakeAnthology(PAPERS), "v1", tmp_path, FakeEmbeddings())

    papers = deepcopy(PAPERS)[1:]
    papers[0] = make_paper(
        "2022.emnlp-1",
        "Neural machine translation again",
        "Revised abstract.",
        ["Bob Ray"],
        2022,
        ["emnlp"],
    )
    papers.append(make_paper("2024.acl-1", "Brand new paper", "Fresh.", ["Eve Fox"], 2024, ["acl"]))

    class Progress:
        def __init__(self, total, desc, unit):
            self.total = total
            self.desc = desc
            self.unit = unit
            self.updates = []

        def update(self, amount):
            self.updates.append(amount)

    progress_bars = []

    @contextmanager
    def track_progress(**kwargs):
        progress = Progress(**kwargs)
        progress_bars.append(progress)
        yield progress

    monkeypatch.setattr("anthology_rag.db.build.tqdm", track_progress)
    build_or_update(FakeAnthology(papers), "v2", tmp_path, FakeEmbeddings())

    assert len(progress_bars) == 1
    progress = progress_bars[0]
    assert progress.desc == "Updating database"
    assert progress.unit == "paper"
    assert progress.total == sum(progress.updates) == 3


def test_max_papers_keeps_most_recent_and_limit_change_rebuilds(tmp_path):
    store = build_or_update(FakeAnthology(PAPERS), "v1", tmp_path, FakeEmbeddings(), max_papers=2)
    assert ids(store) == {"2023.acl-1", "2023.acl-2"}

    # Same version but a different limit must still update the store.
    store = build_or_update(FakeAnthology(PAPERS), "v1", tmp_path, FakeEmbeddings(), max_papers=3)
    assert ids(store) == {"2022.emnlp-1", "2023.acl-1", "2023.acl-2"}

    store = build_or_update(FakeAnthology(PAPERS), "v1", tmp_path, FakeEmbeddings())
    assert len(ids(store)) == 4
