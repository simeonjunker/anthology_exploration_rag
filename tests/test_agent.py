import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from anthology_rag.agent import MAX_HISTORY_MESSAGES, Filters, build_graph
from anthology_rag.db import build_or_update
from tests.fakes import PAPERS, FakeAnthology, FakeEmbeddings


class FakeLLM:
    """Returns queued Filters from the parser and echoes the paper list as the answer."""

    def __init__(self, filters: list[Filters]):
        self.filters = list(filters)
        self.parse_inputs: list[list] = []
        self.respond_inputs: list[list] = []

    def with_structured_output(self, schema):
        def parse(messages):
            self.parse_inputs.append(messages)
            return self.filters.pop(0)

        return RunnableLambda(parse)

    def invoke(self, messages):
        self.respond_inputs.append(messages)
        return AIMessage(messages[0].content)


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


def test_answer_is_grounded_in_retrieved_papers(store):
    llm = FakeLLM([Filters(topic="translation", venues=["emnlp"])])
    result = ask(build_graph(llm, store), "translation at emnlp")
    system_prompt = result["messages"][-1].content
    assert "Neural machine translation again" in system_prompt
    assert "Speech recognition" not in system_prompt


def test_no_match_answers_without_calling_llm(store):
    llm = FakeLLM([Filters(topic="translation", year_from=2030)])
    result = ask(build_graph(llm, store), "translation since 2030")
    assert result["docs"] == []
    assert llm.respond_inputs == []
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
        topic="translation", authors=["Ann Lee"], year_from=2023, year_to=2024, venues=["acl"]
    )
    result = ask(build_graph(FakeLLM([invented]), store), "papers about translation")
    assert result["filters"] == Filters(topic="translation")
    assert len(result["docs"]) == 4  # nothing filtered out


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
