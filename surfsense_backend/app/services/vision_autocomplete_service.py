"""Vision autocomplete service — agent-based with scoped filesystem.

Optimized pipeline:
1. Start the SSE stream immediately so the UI shows progress.
2. Derive a KB search query from window_title (no separate LLM call).
3. Run KB filesystem pre-computation and agent graph compilation in PARALLEL.
4. Inject pre-computed KB files as initial state and stream the agent.
"""

import logging
from collections.abc import AsyncGenerator

from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.autocomplete import create_autocomplete_agent, stream_autocomplete_agent
from app.services.llm_service import get_vision_llm
from app.services.new_streaming_service import VercelStreamingService

logger = logging.getLogger(__name__)

PREP_STEP_ID = "autocomplete-prep"


def _derive_kb_query(app_name: str, window_title: str) -> str:
    parts = [p for p in (window_title, app_name) if p]
    return " ".join(parts)


def _is_vision_unsupported_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "content must be a string" in msg or "does not support image" in msg


async def _extract_query_from_screenshot(
    llm,
    screenshot_data_url: str,
    app_name: str = "",
    window_title: str = "",
) -> str | None:
    """Ask the Vision LLM to describe what the user is working on.

    Raises vision-unsupported errors so the caller can return a
    friendly message immediately instead of retrying with astream.
    """
    if app_name:
        prompt_text = EXTRACT_QUERY_PROMPT_WITH_APP.format(
            app_name=app_name,
            window_title=window_title,
        )
    else:
        prompt_text = EXTRACT_QUERY_PROMPT

    try:
        response = await llm.ainvoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": screenshot_data_url},
                        },
                    ]
                ),
            ]
        )
        query = response.content.strip() if hasattr(response, "content") else ""
        return query if query else None
    except Exception as e:
        if _is_vision_unsupported_error(e):
            raise
        logger.warning(f"Failed to extract query from screenshot: {e}")
        return None


async def _search_knowledge_base(
    session: AsyncSession, search_space_id: int, query: str
) -> str:
    """Search the KB and return formatted context string."""
    try:
        retriever = ChucksHybridSearchRetriever(session)
        results = await retriever.hybrid_search(
            query_text=query,
            top_k=KB_TOP_K,
            search_space_id=search_space_id,
        )

        if not results:
            return ""

        parts: list[str] = []
        char_count = 0
        for doc in results:
            title = doc.get("document", {}).get("title", "Untitled")
            for chunk in doc.get("chunks", []):
                content = chunk.get("content", "").strip()
                if not content:
                    continue
                entry = f"[{title}]\n{content}"
                if char_count + len(entry) > KB_MAX_CHARS:
                    break
                parts.append(entry)
                char_count += len(entry)
            if char_count >= KB_MAX_CHARS:
                break

        return "\n\n---\n\n".join(parts)
    except Exception as e:
        logger.warning(f"KB search failed, proceeding without context: {e}")
        return ""


async def stream_vision_autocomplete(
    screenshot_data_url: str,
    search_space_id: int,
    session: AsyncSession,
    *,
    app_name: str = "",
    window_title: str = "",
) -> AsyncGenerator[str, None]:
    """Analyze a screenshot with a vision-LLM agent and stream a text completion."""
    streaming = VercelStreamingService()
    vision_error_msg = (
        "The selected model does not support vision. "
        "Please set a vision-capable model (e.g. GPT-4o, Gemini) in your search space settings."
    )

    llm = await get_vision_llm(session, search_space_id)
    if not llm:
        yield streaming.format_message_start()
        yield streaming.format_error("No Vision LLM configured for this search space")
        yield streaming.format_done()
        return

    # Start SSE stream immediately so the UI has something to show
    yield streaming.format_message_start()

    kb_query = _derive_kb_query(app_name, window_title)

    # Show a preparation step while KB search + agent compile run
    yield streaming.format_thinking_step(
        step_id=PREP_STEP_ID,
        title="Searching knowledge base",
        status="in_progress",
        items=[kb_query] if kb_query else [],
    )

    try:
        query = await _extract_query_from_screenshot(
            llm,
            screenshot_data_url,
            app_name=app_name,
            window_title=window_title,
        )
    except Exception as e:
        logger.warning(
            f"Vision autocomplete: selected model does not support vision: {e}"
        )
        yield streaming.format_message_start()
        yield streaming.format_error(vision_error_msg)
        yield streaming.format_done()
        return

    has_kb = kb.has_documents
    doc_count = len(kb.files) if has_kb else 0  # type: ignore[arg-type]

    yield streaming.format_thinking_step(
        step_id=PREP_STEP_ID,
        title="Searching knowledge base",
        status="complete",
        items=[f"Found {doc_count} document{'s' if doc_count != 1 else ''}"]
        if kb_query
        else ["Skipped"],
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze this screenshot. Understand the full context of what the user is working on, then generate the text they most likely want to write in the active text area.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": screenshot_data_url},
                },
            ]
        ),
    ]

    user_message = HumanMessage(
        content=[
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": screenshot_data_url}},
        ]
    )

    input_data: dict = {"messages": [user_message]}

    if has_kb:
        input_data["files"] = kb.files
        input_data["messages"] = [kb.ls_ai_msg, kb.ls_tool_msg, user_message]
        logger.info(
            "Autocomplete: injected %d KB files into agent initial state", doc_count
        )
    else:
        logger.info(
            "Autocomplete: no KB documents found, proceeding with screenshot only"
        )

    # Stream the agent (message_start already sent above)
    try:
        async for sse in stream_autocomplete_agent(
            agent,
            input_data,
            streaming,
            emit_message_start=False,
        ):
            yield sse
    except Exception as e:
        if _is_vision_unsupported_error(e):
            logger.warning(
                f"Vision autocomplete: selected model does not support vision: {e}"
            )
            yield streaming.format_error(vision_error_msg)
            yield streaming.format_done()
        else:
            logger.error("Vision autocomplete streaming error: %s", e, exc_info=True)
            yield streaming.format_error("Autocomplete failed. Please try again.")
            yield streaming.format_done()
