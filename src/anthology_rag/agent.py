import re
from typing import Annotated, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from anthology_rag.config import (
    MAX_HISTORY_MESSAGES,
    NUM_CANDIDATES_WITH_AUTHORS,
    NUM_RESULTS,
)

PARSE_PROMPT = (
    "Turn the user's latest request for ACL Anthology papers into a standalone search. "
    "Use the conversation to resolve follow-ups such as 'only from 2023' or 'by the same "
    "authors'. Keep the previous constraints unless the user changes or drops them. "
    "Venues are lowercase anthology venue IDs, e.g. acl, emnlp, naacl, coling, lrec. "
    "Only set authors, years or venues that the user explicitly asked for in the "
    "conversation; never guess or infer them. Leave them empty otherwise."
)
RESPOND_PROMPT = (
    "You help users find papers in the ACL Anthology. Answer using ONLY the papers below. "
    "List the relevant ones as a numbered list with title, authors, venue, year and URL. "
    "Do not invent papers.\n\nRetrieved papers:\n{papers}"
)


class Filters(BaseModel):
    """Structured search request the LLM extracts from the conversation."""

    topic: str = Field(description="Standalone topic to search for")
    authors: list[str] = Field(default_factory=list, description="Author names")
    year_from: int | None = None
    year_to: int | None = None
    venues: list[str] = Field(default_factory=list)


class State(TypedDict, total=False):
    # add_messages appends new messages, so the checkpointer accumulates the chat history.
    messages: Annotated[list[BaseMessage], add_messages]
    # filters and docs are recomputed on every turn.
    filters: Filters
    docs: list[Document]


def _history(messages: list[BaseMessage]) -> list[BaseMessage]:
    # token_counter=len counts messages instead of tokens; start_on="human" avoids
    # beginning the trimmed history with an orphaned AI reply.
    return trim_messages(
        messages,
        max_tokens=MAX_HISTORY_MESSAGES,
        token_counter=len,
        strategy="last",
        start_on="human",
    )


def _only_requested(filters: Filters, messages: list[BaseMessage]) -> Filters:
    """Drop filters the user never wrote: the LLM tends to invent authors, years and venues."""
    text = " ".join(str(m.content) for m in messages if m.type == "human").lower()
    words = set(re.findall(r"\w+", text))
    return filters.model_copy(
        update={
            "authors": [
                a for a in filters.authors if all(p in words for p in re.findall(r"\w+", a.lower()))
            ],
            "venues": [v for v in filters.venues if v.lower() in words],
            "year_from": filters.year_from
            if filters.year_from and str(filters.year_from) in words
            else None,
            "year_to": filters.year_to
            if filters.year_to and str(filters.year_to) in words
            else None,
        }
    )


def _where(filters: Filters) -> dict | None:
    """Translate year/venue filters into a Chroma `where` clause (None = no filter)."""
    clauses: list[dict] = []
    if filters.year_from is not None:
        clauses.append({"year": {"$gte": filters.year_from}})
    if filters.year_to is not None:
        clauses.append({"year": {"$lte": filters.year_to}})
    # Matches the per-venue flags written by the db component; any listed venue may match.
    venue_clauses = [{f"venue_{v.lower()}": True} for v in filters.venues]
    if len(venue_clauses) == 1:
        clauses.append(venue_clauses[0])
    elif venue_clauses:
        clauses.append({"$or": venue_clauses})
    # Chroma rejects "$and" with fewer than two conditions.
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _format_paper(doc: Document) -> str:
    m = doc.metadata
    return (
        f"- {m['title']} | authors: {m['authors']} | venue: {m['venues']} | "
        f"year: {m['year']} | url: {m['url']}\n  abstract: {m['abstract'][:500]}"
    )


def build_graph(llm: BaseChatModel, store: Chroma) -> CompiledStateGraph:
    parser = llm.with_structured_output(Filters)

    # Node 1 (LLM): the whole history goes in, so follow-ups become standalone searches.
    def parse_query(state: State) -> State:
        history = _history(state["messages"])
        filters = parser.invoke([SystemMessage(PARSE_PROMPT), *history])
        return {"filters": _only_requested(filters, history)}

    # Node 2 (no LLM): vector search on the topic, metadata filters applied in Chroma.
    def retrieve(state: State) -> State:
        filters = state["filters"]
        k = NUM_CANDIDATES_WITH_AUTHORS if filters.authors else NUM_RESULTS
        docs = store.similarity_search(filters.topic, k=k, filter=_where(filters))
        for author in filters.authors:
            docs = [d for d in docs if author.lower() in d.metadata["authors"].lower()]
        return {"docs": docs[:NUM_RESULTS]}

    # Node 3 (LLM): write the answer from the retrieved papers; the reply is added to the
    # history, but the papers themselves are not, so old results don't fill the context.
    def respond(state: State) -> State:
        docs = state["docs"]
        if not docs:
            return {"messages": [AIMessage("No papers matched your request.")]}
        papers = "\n".join(_format_paper(d) for d in docs)
        prompt = SystemMessage(RESPOND_PROMPT.format(papers=papers))
        return {"messages": [llm.invoke([prompt, *_history(state["messages"])])]}

    graph = StateGraph(State)
    graph.add_node("parse_query", parse_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("respond", respond)
    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "retrieve")
    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=InMemorySaver())
