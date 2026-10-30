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
    MAX_RESULTS,
    MIN_RESULTS,
    MIN_SIMILARITY,
    NUM_CANDIDATES_WITH_AUTHORS,
)

PARSE_PROMPT = (
    "Turn the user's latest request for ACL Anthology papers into a standalone search. "
    "Use the conversation to resolve follow-ups such as 'only from 2023' or 'by the same "
    "authors'. Keep the previous constraints unless the user changes or drops them. "
    "Venues are lowercase anthology venue IDs, e.g. acl, emnlp, naacl, coling, lrec. "
    "Only set authors, years or venues that the user explicitly asked for in the "
    "conversation; never guess or infer them. Leave them empty otherwise. "
    "If the latest message is not a request for papers (thanks, greetings, small talk, "
    "questions about you), set reply to a short, friendly answer and leave the search "
    "fields empty; otherwise leave reply empty. "
    "Set num_results only if the user asks for a specific number of papers. "
    "Set more=true when the user asks for more or further results of the same search "
    "(e.g. 'more', 'show me 5 more'); keep the previous topic and constraints then. "
    "Set similar_to to the list number when the user asks for papers similar to a listed "
    "paper (e.g. 'similar papers like 15'); the topic can then be that paper's title."
)


class Filters(BaseModel):
    """Structured search request the LLM extracts from the conversation."""

    topic: str = Field(default="", description="Standalone topic to search for")
    reply: str = Field(default="", description="Answer to a message that is not a paper search")
    authors: list[str] = Field(default_factory=list, description="Author names")
    year_from: int | None = None
    year_to: int | None = None
    venues: list[str] = Field(default_factory=list)
    num_results: int | None = Field(default=None, ge=1, description="Requested number of papers")
    more: bool = Field(default=False, description="User wants further results of the last search")
    similar_to: int | None = Field(
        default=None, ge=1, description="Number of a listed paper the user wants similar ones to"
    )


class State(TypedDict, total=False):
    # add_messages appends new messages, so the checkpointer accumulates the chat history.
    messages: Annotated[list[BaseMessage], add_messages]
    # filters, docs and scores are recomputed on every turn; scores[i] belongs to docs[i].
    filters: Filters
    docs: list[Document]
    scores: list[float]
    # Papers of the current search shown so far; "more" continues after them, and
    # "similar to N" looks up paper N in `listed` (numbers match what the user sees).
    shown: int
    listed: list[Document]
    # Paper a similar-paper search is based on, kept so "more" continues that search.
    anchor: Document | None
    error: str
    # True when a "more" request changed the constraints, so results start again at 1.
    restart: bool
    # True when the last message was not a search; the previous search stays untouched.
    is_chat: bool


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
            "similar_to": filters.similar_to
            if filters.similar_to and str(filters.similar_to) in words
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


def _format_paper(rank: int, doc: Document, score: float) -> str:
    m = doc.metadata
    return (
        f"{rank}. {m['title']}\n"
        f"   Authors: {m['authors']}\n"
        f"   Year: {m['year']} | Venue: {m['venues']} | Score: {score:.2f}\n"
        f"   Link: {m['url']}"
    )


def build_graph(llm: BaseChatModel, store: Chroma) -> CompiledStateGraph:
    parser = llm.with_structured_output(Filters)

    # Node 1 (LLM): the whole history goes in, so follow-ups become standalone searches.
    def parse_query(state: State) -> State:
        history = _history(state["messages"])
        filters = parser.invoke([SystemMessage(PARSE_PROMPT), *history])
        if filters.reply:
            return {"messages": [AIMessage(filters.reply)], "is_chat": True}
        if filters.more and "filters" in state:
            # "More" keeps the previous topic and similar-paper anchor; re-deriving them from
            # the history made the LLM drift. Only constraints the user states anew (e.g.
            # "from 2025 or later") are taken from the LLM; if they change the search, it
            # restarts from the top instead of continuing the old result list.
            previous = state["filters"]
            requested = _only_requested(filters, history)
            constraints = {
                "authors": requested.authors or previous.authors,
                "venues": requested.venues or previous.venues,
                "year_from": requested.year_from or previous.year_from,
                "year_to": requested.year_to or previous.year_to,
            }
            restart = any(getattr(previous, key) != value for key, value in constraints.items())
            update = {"more": True, "num_results": filters.num_results, "similar_to": None}
            return {
                "filters": previous.model_copy(update=update | constraints),
                "restart": restart,
                "is_chat": False,
            }
        return {"filters": _only_requested(filters, history), "restart": False, "is_chat": False}

    # Node 2 (no LLM): vector search on the topic, metadata filters applied in Chroma.
    # Explicit count or "more": a page of that size after the papers already shown.
    # Otherwise: everything above MIN_SIMILARITY, within MIN_RESULTS..MAX_RESULTS.
    def retrieve(state: State) -> State:
        filters = state["filters"]
        listed = state.get("listed", [])
        continuing = filters.more and not state["restart"]
        anchor = state.get("anchor") if filters.more else None
        if filters.similar_to is not None:
            if not 1 <= filters.similar_to <= len(listed):
                return {
                    "docs": [],
                    "scores": [],
                    "shown": 0,
                    "error": f"There is no paper {filters.similar_to} in the last list.",
                }
            anchor = listed[filters.similar_to - 1]
        offset = state.get("shown", 0) if continuing else 0
        page = filters.num_results or MAX_RESULTS
        k = offset + page + (anchor is not None)  # the anchor paper itself is dropped below
        if filters.authors:
            k = max(k, NUM_CANDIDATES_WITH_AUTHORS)
        # Similar-paper search embeds the anchor's title and abstract (its page_content).
        query = anchor.page_content if anchor else filters.topic
        hits = store.similarity_search_with_score(query, k=k, filter=_where(filters))
        if anchor:
            hits = [(d, s) for d, s in hits if d.id != anchor.id]
        for author in filters.authors:
            hits = [(d, s) for d, s in hits if author.lower() in d.metadata["authors"].lower()]
        # The collection uses cosine distance, so similarity = 1 - distance (1.0 = identical).
        hits = [(d, 1 - s) for d, s in hits]
        if continuing or filters.num_results:
            hits = hits[offset : offset + page]
        else:
            relevant = sum(score >= MIN_SIMILARITY for _, score in hits)
            hits = hits[: min(MAX_RESULTS, max(relevant, MIN_RESULTS))]
        docs = [d for d, _ in hits]
        return {
            "docs": docs,
            "scores": [score for _, score in hits],
            "shown": offset + len(hits),
            "listed": listed + docs if continuing else docs,
            "anchor": anchor,
            "error": "",
        }

    # Node 3 (no LLM): a fixed format keeps every answer identical in structure and means the
    # scores are exactly the ones Chroma computed. The list is added to the chat history.
    def respond(state: State) -> State:
        if state["error"]:
            return {"messages": [AIMessage(state["error"])]}
        if not state["docs"]:
            text = "No more papers found." if state["shown"] else "No papers matched your request."
            return {"messages": [AIMessage(text)]}
        first_rank = state["shown"] - len(state["docs"]) + 1
        papers = [
            _format_paper(i, d, s)
            for i, (d, s) in enumerate(zip(state["docs"], state["scores"]), start=first_rank)
        ]
        return {"messages": [AIMessage("\n\n".join(papers))]}

    graph = StateGraph(State)
    graph.add_node("parse_query", parse_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("respond", respond)
    graph.add_edge(START, "parse_query")
    graph.add_conditional_edges(
        "parse_query", lambda state: END if state["is_chat"] else "retrieve"
    )
    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=InMemorySaver())
