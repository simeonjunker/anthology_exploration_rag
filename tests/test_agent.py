import pytest
from langchain_core.runnables import RunnableLambda

from anthology_rag.agent import MAX_HISTORY_MESSAGES, Filters, build_graph
from anthology_rag.db import build_or_update
from tests.fakes import PAPERS, FakeAnthology, FakeEmbeddings


class FakeLLM:
    """Returns queued Filters from the parser; the answer is formatted without the LLM."""

    def __init__(self, filters: list[Filters]):
        self.filters = list(filters)
        self.parse_inputs: list[list] = []

    def with_structured_output(self, schema):
        def parse(messages):
            self.parse_inputs.append(messages)
            return self.filters.pop(0)

        return RunnableLambda(parse)


@pytest.fixture
def store(tmp_path):
    return build_or_update(FakeAnthology(PAPERS), "v1", tmp_path, FakeEmbeddings())


def ask(graph, text, thread="t"):
    return graph.invoke({"messages": [("user", text)]}, {"configurable": {"thread_id": thread}})


def titles(result):
    return {d.metadata["title"] for d in result["docs"]}


def test_topic_search_ranks_relevant_paper_first(store):
    graph = build_graph(FakeLLM([Filters(topic="speech recognition")]), store)
    result = ask(graph, "speech papers")
    assert result["docs"][0].metadata["title"] == "Speech recognition with transformers"


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        (
            Filters(topic="translation", year_from=2022),
            {"2022.emnlp-1", "2023.acl-1", "2023.acl-2"},
        ),
        (Filters(topic="translation", year_from=2021, year_to=2022), {"2022.emnlp-1"}),
        (Filters(topic="translation", year_to=2020), {"2020.acl-1"}),
        (Filters(topic="translation", venues=["EMNLP"]), {"2022.emnlp-1"}),
        (
            Filters(topic="translation", venues=["emnlp", "ws"]),
            {"2022.emnlp-1", "2023.acl-1"},
        ),
        (
            Filters(topic="translation", authors=["ann lee"], year_from=2021),
            {"2023.acl-1"},
        ),
        (
            Filters(topic="translation", authors=["Ann Lee"]),
            {"2020.acl-1", "2023.acl-1"},
        ),
        (
            Filters(topic="translation", venues=["acl"], year_from=2023, authors=["Cy Dee"]),
            {"2023.acl-2"},
        ),
    ],
)
def test_filters_return_exactly_matching_papers(store, filters, expected_ids):
    question = "translation from 2020 2021 2022 2023 2030 EMNLP ws acl by Ann Lee and Cy Dee"
    result = ask(build_graph(FakeLLM([filters]), store), question)
    assert {d.id for d in result["docs"]} == expected_ids


def test_answer_lists_title_authors_year_venue_and_score(store):
    llm = FakeLLM([Filters(topic="translation", venues=["emnlp"])])
    result = ask(build_graph(llm, store), "translation at emnlp")
    answer = result["messages"][-1].content
    doc = result["docs"][0]
    assert doc.metadata["title"] == "Neural machine translation again"
    assert answer.startswith(f"1. {doc.metadata['title']}")
    assert f"Authors: {doc.metadata['authors']}" in answer
    assert f"Year: {doc.metadata['year']} | Venue: {doc.metadata['venues']}" in answer
    assert f"Score: {result['scores'][0]:.2f}" in answer
    assert f"Link: {doc.metadata['url']}" in answer
    assert "Speech recognition" not in answer


def test_scores_are_similarities_ordered_best_first(store):
    result = ask(build_graph(FakeLLM([Filters(topic="speech recognition")]), store), "speech")
    scores = result["scores"]
    assert len(scores) == len(result["docs"])
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_no_match_answers_without_papers(store):
    llm = FakeLLM([Filters(topic="translation", year_from=2030)])
    result = ask(build_graph(llm, store), "translation since 2030")
    assert result["docs"] == []
    assert "No papers" in result["messages"][-1].content


def test_history_persists_per_thread_and_reaches_parser(store):
    llm = FakeLLM([Filters(topic="translation"), Filters(topic="translation", year_from=2023)] * 2)
    graph = build_graph(llm, store)
    ask(graph, "translation papers")
    second = ask(graph, "only from 2023")
    assert len(second["messages"]) == 4
    parser_roles = [m.type for m in llm.parse_inputs[1]]
    assert parser_roles == ["system", "human", "ai", "human"]
    assert llm.parse_inputs[1][1].content == "translation papers"

    fresh = ask(graph, "hello", thread="other")
    assert len(fresh["messages"]) == 2


def test_history_is_trimmed(store):
    n_turns = MAX_HISTORY_MESSAGES
    llm = FakeLLM([Filters(topic="translation")] * n_turns)
    graph = build_graph(llm, store)
    for i in range(n_turns):
        ask(graph, f"question {i}")
    sent = llm.parse_inputs[-1][1:]  # without system prompt
    assert len(sent) <= MAX_HISTORY_MESSAGES
    assert sent[0].type == "human"
    assert sent[-1].content == f"question {n_turns - 1}"


def test_filters_not_mentioned_by_user_are_dropped(store):
    invented = Filters(
        topic="translation",
        authors=["Ann Lee"],
        year_from=2023,
        year_to=2024,
        venues=["acl"],
        num_results=10,
    )
    result = ask(build_graph(FakeLLM([invented]), store), "papers about translation")
    assert result["filters"] == Filters(topic="translation", num_results=10)
    assert len(result["docs"]) == 4  # nothing filtered out


def test_explicit_count_limits_results(store):
    graph = build_graph(FakeLLM([Filters(topic="translation", num_results=2)]), store)
    result = ask(graph, "2 papers about translation")
    assert len(result["docs"]) == 2
    assert result["scores"] == sorted(result["scores"], reverse=True)


def test_default_count_stays_within_min_and_max(store, monkeypatch):
    import anthology_rag.agent as agent

    monkeypatch.setattr(agent, "MIN_SIMILARITY", 2.0)  # nothing is similar enough
    monkeypatch.setattr(agent, "MIN_RESULTS", 2)
    result = ask(build_graph(FakeLLM([Filters(topic="translation")]), store), "translation")
    assert len(result["docs"]) == 2

    monkeypatch.setattr(agent, "MIN_SIMILARITY", -1.0)  # everything is similar enough
    monkeypatch.setattr(agent, "MAX_RESULTS", 3)
    result = ask(build_graph(FakeLLM([Filters(topic="translation")]), store), "translation")
    assert len(result["docs"]) == 3


def test_default_returns_all_papers_above_threshold(store, monkeypatch):
    import anthology_rag.agent as agent

    monkeypatch.setattr(agent, "MIN_RESULTS", 1)
    monkeypatch.setattr(agent, "MIN_SIMILARITY", 0.5)
    result = ask(build_graph(FakeLLM([Filters(topic="speech recognition")]), store), "speech")
    assert len(result["docs"]) == sum(s >= 0.5 for s in result["scores"]) >= 1


def test_more_continues_after_shown_papers(store):
    llm = FakeLLM(
        [
            Filters(topic="translation", num_results=2),
            Filters(topic="translation", more=True),
            Filters(topic="translation", more=True),
        ]
    )
    graph = build_graph(llm, store)
    first = ask(graph, "2 papers about translation")
    second = ask(graph, "more")
    assert {d.id for d in first["docs"]}.isdisjoint(d.id for d in second["docs"])
    assert len(second["docs"]) == 2
    assert second["messages"][-1].content.startswith("3. ")
    third = ask(graph, "more")
    assert third["messages"][-1].content == "No more papers found."


def test_earlier_turn_constraints_are_kept_in_follow_ups(store):
    llm = FakeLLM(
        [
            Filters(topic="translation", year_from=2022),
            Filters(topic="translation", year_from=2022, venues=["emnlp"]),
        ]
    )
    graph = build_graph(llm, store)
    ask(graph, "translation papers since 2022")
    result = ask(graph, "only at EMNLP")
    assert result["filters"].year_from == 2022
    assert result["filters"].venues == ["emnlp"]


def list_all_papers(graph):
    return ask(graph, "all 4 translation papers", thread="s")


def test_similar_to_uses_listed_paper_text_and_excludes_it(store):
    llm = FakeLLM(
        [
            Filters(topic="translation", num_results=4),
            Filters(topic="x", similar_to=2, num_results=3),
        ]
    )
    graph = build_graph(llm, store)
    first = list_all_papers(graph)
    anchor = first["docs"][1]
    second = ask(graph, "similar papers like 2", thread="s")
    expected = store.similarity_search_with_score(anchor.page_content, k=4)
    expected = [d.id for d, _ in expected if d.id != anchor.id]
    assert [d.id for d in second["docs"]] == expected
    assert anchor.id not in expected
    assert second["messages"][-1].content.startswith("1. ")
    assert second["scores"] == sorted(second["scores"], reverse=True)


def test_similar_to_unknown_number_gives_message(store):
    llm = FakeLLM([Filters(topic="translation", num_results=2), Filters(topic="x", similar_to=9)])
    graph = build_graph(llm, store)
    ask(graph, "2 translation papers", thread="s")
    result = ask(graph, "similar to 9", thread="s")
    assert result["messages"][-1].content == "There is no paper 9 in the last list."


def test_more_after_similar_continues_similar_search(store):
    llm = FakeLLM(
        [
            Filters(topic="translation", num_results=4),
            Filters(topic="x", similar_to=1, num_results=1),
            Filters(topic="x", more=True, num_results=1),
        ]
    )
    graph = build_graph(llm, store)
    first = list_all_papers(graph)
    second = ask(graph, "similar to 1", thread="s")
    third = ask(graph, "more", thread="s")
    ids = [d.id for d in second["docs"] + third["docs"]]
    assert len(set(ids)) == 2 and first["docs"][0].id not in ids
    assert third["messages"][-1].content.startswith("2. ")


def test_more_reuses_previous_search_and_ignores_llm_filters(store):
    llm = FakeLLM(
        [
            Filters(topic="translation", num_results=4),
            Filters(topic="x", similar_to=1, num_results=1),
            # What a model might emit for "more": drifted topic, filters and a stale similar_to.
            Filters(topic="speech", year_from=2023, venues=["acl"], more=True, similar_to=1),
        ]
    )
    graph = build_graph(llm, store)
    list_all_papers(graph)
    second = ask(graph, "similar to 1", thread="s")
    third = ask(graph, "more", thread="s")
    assert third["filters"].topic == "x"
    assert third["filters"].year_from is None and third["filters"].venues == []
    assert third["filters"].similar_to is None
    assert third["messages"][-1].content.startswith("2. ")
    assert not {d.id for d in second["docs"]} & {d.id for d in third["docs"]}


def test_new_constraint_on_follow_up_restarts_similar_search(store):
    llm = FakeLLM(
        [
            Filters(topic="translation", num_results=4),
            Filters(topic="x", similar_to=1, num_results=2),
            # The model flags a pure filter follow-up as "more" and drifts the topic.
            Filters(topic="speech", year_from=2022, more=True),
        ]
    )
    graph = build_graph(llm, store)
    first = list_all_papers(graph)
    ask(graph, "similar to 1", thread="s")
    third = ask(graph, "from 2022 or later", thread="s")
    assert third["filters"].year_from == 2022
    assert third["filters"].topic == "x"  # still the similar-paper search
    assert third["messages"][-1].content.startswith("1. ")  # restarted, not continued
    assert all(d.metadata["year"] >= 2022 for d in third["docs"])
    assert first["docs"][0].id not in {d.id for d in third["docs"]}
    assert third["docs"]
