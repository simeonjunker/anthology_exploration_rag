import uuid

from langchain_openai import ChatOpenAI

from anthology_rag.agent import build_graph
from anthology_rag.config import (
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    PERSIST_DIR,
    ENABLE_THINKING,
)
from anthology_rag.db import get_embeddings, get_store


def main() -> None:
    if not (PERSIST_DIR / "meta.json").exists():
        raise SystemExit(f"No database in {PERSIST_DIR}; run `uv run build-db` first.")
    # Local llama.cpp/vLLM server speaking the OpenAI API; it ignores the API key.
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key="none",
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        extra_body={"chat_template_kwargs": {"enable_thinking": ENABLE_THINKING}},
    )
    graph = build_graph(llm, get_store(PERSIST_DIR, get_embeddings()))

    # The checkpointer keys the history by thread_id, so a new id starts a fresh chat.
    thread_id = str(uuid.uuid4())
    print("Ask for papers. /reset clears the conversation, /quit exits.")
    while True:
        question = input("> ").strip()
        if question == "/quit":
            break
        if question == "/reset":
            thread_id = str(uuid.uuid4())
            print("Conversation reset.")
        elif question:
            # Only the new message is passed in; earlier turns come from the checkpointer.
            result = graph.invoke(
                {"messages": [("user", question)]},
                {"configurable": {"thread_id": thread_id}},
            )
            if not result["is_chat"]:
                print(f"[search] {result['filters'].model_dump(exclude_defaults=True)}\n")
            print(result["messages"][-1].content)
