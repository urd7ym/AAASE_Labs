#!/usr/bin/env python3
# ============================================================
# DAY 3 LAB — From Prototype to Enterprise
# ============================================================

from __future__ import annotations

import json
import logging
import operator
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

try:
    from fastapi import Request
except ImportError:  # pragma: no cover
    Request = Any

load_dotenv()

STAGE = int(os.getenv("LAB_STAGE", "0"))
MOCK = os.getenv("MOCK", "0") == "1"


class FakeResponse:
    def __init__(self, content: str):
        self.content = content
        self.usage_metadata = {"input_tokens": 200, "output_tokens": 300}


class FakeChatModel:
    def __init__(self):
        self.review_calls = 0

    def invoke(self, prompt: str):
        time.sleep(0.2)
        p = prompt.lower()
        if "reviewer" in p:
            self.review_calls += 1
            score = 6 if self.review_calls == 1 else 9
            return FakeResponse(f"SCORE: {score}\nFEEDBACK: Add a concrete example.")
        if "research" in p:
            return FakeResponse("- fact one\n- fact two\n- fact three")
        if "summar" in p:
            return FakeResponse("A concise summary of the research notes.")
        return FakeResponse("INTRODUCTION\n...\n\nBODY\n" + "Substantive findings. " * 20 + "\n\nCONCLUSION\n...")


class ReportState(TypedDict, total=False):
    run_id: str
    topic: str
    research_notes: str
    summary: str
    draft: str
    review_feedback: str
    score: int
    revision_count: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error: str
    execution_logs: Annotated[List[str], operator.add]


@dataclass
class Settings:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    request_timeout_s: float = 60.0
    max_retries: int = 2
    quality_threshold: int = 8
    max_revisions: int = 2
    cost_budget_usd: float = 0.05
    max_topic_len: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            request_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "60")),
            max_retries=int(os.getenv("MAX_RETRIES", "2")),
            quality_threshold=int(os.getenv("QUALITY_THRESHOLD", "8")),
            max_revisions=int(os.getenv("MAX_REVISIONS", "2")),
            cost_budget_usd=float(os.getenv("COST_BUDGET_USD", "0.05")),
            max_topic_len=int(os.getenv("MAX_TOPIC_LEN", "120")),
        )


if MOCK:
    model = FakeChatModel()
else:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install langchain-openai to use non-mock mode") from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set MOCK=1 or provide OPENAI_API_KEY")
    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        timeout=60,
        max_retries=0,
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL") or None,
    )

SETTINGS = Settings.from_env() if STAGE >= 2 else Settings()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        if hasattr(record, "payload"):
            payload.update(record.payload)
        return json.dumps(payload, default=str)


logger = logging.getLogger("enterprise_lab")
logger.setLevel(logging.INFO)
logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.propagate = False


def log_event(event: str, **fields: Any) -> None:
    logger.info(event, extra={"payload": fields})


class BudgetExceeded(Exception):
    pass


def call_llm(prompt: str, node: str, state: ReportState) -> str:
    if STAGE >= 4 and state.get("cost_usd", 0.0) >= SETTINGS.cost_budget_usd:
        raise BudgetExceeded("cost budget exhausted")

    start = time.perf_counter()
    max_retries = SETTINGS.max_retries if STAGE >= 1 else 0
    for attempt in range(1, max_retries + 2):
        try:
            if isinstance(model, FakeChatModel):
                response = model.invoke(prompt)
            else:
                response = model.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            usage = getattr(response, "usage_metadata", {}) or {}
            tokens_in = int(usage.get("input_tokens", len(prompt.split())))
            tokens_out = int(usage.get("output_tokens", len(content.split())))
            cost = (tokens_in + tokens_out) * 0.000002
            state["tokens_in"] = int(state.get("tokens_in", 0)) + tokens_in
            state["tokens_out"] = int(state.get("tokens_out", 0)) + tokens_out
            state["cost_usd"] = round(float(state.get("cost_usd", 0.0)) + cost, 6)
            if STAGE >= 3:
                log_event(
                    "llm_call",
                    run_id=state.get("run_id"),
                    node=node,
                    attempt=attempt,
                    latency_s=round(time.perf_counter() - start, 3),
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=state["cost_usd"],
                )
            return content
        except Exception as exc:
            if attempt > max_retries:
                raise RuntimeError(f"{node} failed: {exc}") from exc
            delay = 2 ** (attempt - 1) + random.uniform(0, 0.5)
            if STAGE >= 3:
                log_event(
                    "llm_retry",
                    run_id=state.get("run_id"),
                    node=node,
                    attempt=attempt,
                    delay_s=round(delay, 3),
                    error=str(exc),
                )
            time.sleep(delay)

    raise RuntimeError("unreachable")


def validate_topic(topic: str) -> str:
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("topic is required")
    if len(topic) > SETTINGS.max_topic_len:
        raise ValueError("topic is too long")
    patterns = [r"ignore (all|previous|the) instructions", r"system prompt", r"act as"]
    if any(re.search(pattern, topic, re.I) for pattern in patterns):
        raise ValueError("topic appears to be an injection attempt")
    return topic


def validate_report(report: str) -> None:
    if not report:
        raise ValueError("report is empty")
    if len(report.strip()) < 40:
        raise ValueError("report is too short")
    refusal_patterns = [r"as an ai language model", r"i can'?t help", r"cannot help", r"refuse"]
    if any(re.search(pattern, report, re.I) for pattern in refusal_patterns):
        raise ValueError("report contains a refusal artifact")


def research_node(state: ReportState) -> Dict[str, Any]:
    prompt = f"Researcher: gather facts about {state['topic']}"
    research_notes = call_llm(prompt, "research", state)
    return {"research_notes": research_notes}


def summarize_node(state: ReportState) -> Dict[str, Any]:
    prompt = f"Summarize these research notes for {state['topic']}: {state['research_notes']}"
    summary = call_llm(prompt, "summarize", state)
    return {"summary": summary}


def write_node(state: ReportState) -> Dict[str, Any]:
    prompt = f"Write a report about {state['topic']} based on this summary: {state['summary']}"
    if state.get("review_feedback"):
        prompt += f"\nA reviewer said: {state['review_feedback']}. Address this feedback."
    draft = call_llm(prompt, "write", state)
    return {"draft": draft}


def review_node(state: ReportState) -> Dict[str, Any]:
    prompt = (
        f"Reviewer: score the draft from 1-10 and give feedback. "
        f"Topic: {state['topic']}\nDraft: {state['draft']}"
    )
    review_text = call_llm(prompt, "review", state)
    score_match = re.search(r"SCORE:\s*(\d+)", review_text, re.I)
    score = int(score_match.group(1)) if score_match else 0
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", review_text, re.I)
    feedback = feedback_match.group(1).strip() if feedback_match else "Add more detail."
    revision_count = int(state.get("revision_count", 0)) + 1
    if STAGE >= 3:
        log_event("review_verdict", run_id=state.get("run_id"), score=score, revision_count=revision_count)
    return {"review_feedback": feedback, "score": score, "revision_count": revision_count}


def review_gate(state: ReportState) -> str:
    score = int(state.get("score", 0) or 0)
    revision_count = int(state.get("revision_count", 0) or 0)
    if score >= SETTINGS.quality_threshold:
        return "approve"
    if revision_count > SETTINGS.max_revisions:
        return "give_up"
    return "revise"


workflow = StateGraph(ReportState)
workflow.add_node("research", research_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("write", write_node)
workflow.add_node("review", review_node)
workflow.add_edge(START, "research")
workflow.add_edge("research", "summarize")
workflow.add_edge("summarize", "write")
workflow.add_edge("write", "review")
workflow.add_conditional_edges("review", review_gate, {"approve": END, "give_up": END, "revise": "write"})
graph = workflow.compile()


def generate_report(topic: str) -> ReportState:
    state: ReportState = {
        "run_id": str(uuid.uuid4())[:8],
        "topic": topic,
        "research_notes": "",
        "summary": "",
        "draft": "",
        "review_feedback": "",
        "score": 0,
        "revision_count": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "error": "",
        "execution_logs": [],
    }
    if STAGE >= 4:
        try:
            state["topic"] = validate_topic(topic)
        except ValueError as exc:
            state["error"] = str(exc)
            if STAGE >= 3:
                log_event("run_finished", run_id=state["run_id"], error=str(exc), stage=STAGE)
            return state
    if STAGE >= 3:
        log_event("run_started", run_id=state["run_id"], topic=state["topic"], stage=STAGE)
    try:
        final_state = graph.invoke(state)
    except BudgetExceeded as exc:
        state["error"] = str(exc)
        if STAGE >= 3:
            log_event("run_finished", run_id=state["run_id"], error=str(exc), stage=STAGE)
        return state
    except RuntimeError as exc:
        state["error"] = str(exc)
        if STAGE >= 1 and STAGE >= 3:
            log_event("run_finished", run_id=state["run_id"], error=str(exc), stage=STAGE)
        if STAGE >= 1:
            return state
        raise

    if STAGE >= 4:
        validate_report(final_state.get("draft", ""))
    if STAGE >= 3:
        log_event(
            "run_finished",
            run_id=state["run_id"],
            score=final_state.get("score", 0),
            cost_usd=final_state.get("cost_usd", 0.0),
            tokens_in=final_state.get("tokens_in", 0),
            tokens_out=final_state.get("tokens_out", 0),
            stage=STAGE,
        )
    return final_state


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install fastapi and uvicorn for stage 5") from exc

    app = FastAPI(title="Enterprise Multi-Agent Lab")

    @app.get("/health")
    def health():
        return {"status": "ok", "stage": STAGE, "model": "mock" if MOCK else SETTINGS.model_name}

    @app.post("/report")
    async def report(request: Request):
        import json
        body = await request.body()
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            raise HTTPException(status_code=422, detail="request body must be valid JSON") from None
        topic = payload.get("topic", "")
        try:
            result = generate_report(topic)
            if result.get("error"):
                if str(result["error"]).startswith("topic"):
                    raise HTTPException(status_code=422, detail=result["error"])
                raise HTTPException(status_code=503, detail=result["error"])
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


if __name__ == "__main__":
    print(f"=== STAGE {STAGE} {'(MOCK)' if MOCK else ''} ===")
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn

        uvicorn.run(create_app(), host="127.0.0.1", port=8000)
    else:
        topic = os.getenv("TOPIC", "Smart Cities")
        result = generate_report(topic)
        print(json.dumps({"topic": result.get("topic"), "score": result.get("score"), "cost_usd": result.get("cost_usd"), "error": result.get("error")}, indent=2))
        print("\nREPORT")
        print(result.get("draft", ""))
