# ============================================================
# ENTERPRISE AUTONOMOUS RESEARCH AI AGENT  (Completed Version)
# ============================================================

import os
import operator
import itertools
from datetime import datetime
from typing import Annotated, List, Dict
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# ============================================================
# CONFIGURATION
# ============================================================
load_dotenv()
FAKE_MODE = os.getenv("USE_FAKE", "0") == "1"
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_RESEARCH_ITERATIONS = 3
QUALITY_THRESHOLD = 7

# ============================================================
# MODELS, SEARCH, EMBEDDINGS
# ============================================================
if FAKE_MODE:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.embeddings import DeterministicFakeEmbedding

    analysis_llm = GenericFakeChatModel(messages=itertools.cycle([
        AIMessage(content="1. Summary: Agentic AI adoption.\n2. Importance Score: 7\n3. Business Impact: High.")
    ]))
    report_llm = GenericFakeChatModel(messages=itertools.cycle([
        AIMessage(content="# Enterprise Research Report\n\n## Executive Summary\nFake mode executed successfully.")
    ]))
    _fake_scores = iter([4, 9, 9, 9])
    embedding_model = DeterministicFakeEmbedding(size=384)

    def run_search(query: str) -> List[Dict]:
        return [
            {"url": "https://example.com/ai", "title": "AI 2026", "content": f"Results for {query}"}
        ]
else:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_tavily import TavilySearch

    _llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free", 
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
#        model="nvidia/nemotron-3-super-120b-a12b:free",
#        temperature=0,
#        base_url="https://openrouter.ai/api/v1",
    analysis_llm = _llm
    report_llm = _llm
    embedding_model = OpenAIEmbeddings()
    _search_tool = TavilySearch(max_results=5)

    def run_search(query: str) -> List[Dict]:
        response = _search_tool.invoke({"query": query})
        if isinstance(response, dict):
            return response.get("results", [])
        return response

try:
    from langchain_chroma import Chroma
    vector_store = Chroma(
        collection_name="enterprise_research_memory",
        embedding_function=embedding_model,
        persist_directory="./enterprise_memory_db",
    )
except ImportError:
    from langchain_core.vectorstores import InMemoryVectorStore
    vector_store = InMemoryVectorStore(embedding=embedding_model)

# ============================================================
# STEP 3 — STRUCTURED OUTPUT
# ============================================================
class QualityScore(BaseModel):
    """Evaluation of research quality."""
    score: int = Field(ge=1, le=10)
    reasoning: str = Field(description="One-sentence justification")

def evaluate_quality(analyzed_data) -> QualityScore:
    if FAKE_MODE:
        return QualityScore(score=next(_fake_scores), reasoning="Scripted score (fake mode).")
    evaluator = analysis_llm.with_structured_output(QualityScore)
    return evaluator.invoke([HumanMessage(content=f"Evaluate the overall quality of this research on a 1-10 scale.\n\nResearch:\n{analyzed_data}")])

# ============================================================
# STEP 1 — THE STATE
# ============================================================
class AgentState(TypedDict):
    topic: str
    search_query: str
    collected_data: List[Dict]
    analyzed_data: List[Dict]
    quality_score: int
    iteration_count: int
    final_report: str
    execution_logs: Annotated[List[str], operator.add]

def log(message: str) -> List[str]:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line)
    return [line]

# ============================================================
# STEP 4 — NODES
# ============================================================
def collect_node(state: AgentState):
    iteration = state["iteration_count"] + 1
    refinements = [
        state["topic"],
        f"{state['topic']} latest developments case studies",
        f"{state['topic']} industry analysis best practices",
    ]
    query = refinements[min(iteration - 1, len(refinements) - 1)]
    results = run_search(query)
    return {
        "search_query": query,
        "collected_data": results,
        "iteration_count": iteration,
        "execution_logs": log(f"Iteration {iteration}: collected {len(results)} sources for query: '{query}'"),
    }

def store_memory_node(state: AgentState):
    documents = [item.get("content", "") for item in state["collected_data"] if item.get("content")]
    if documents:
        vector_store.add_texts(documents)
    return {"execution_logs": log(f"Stored {len(documents)} documents in vector memory.")}

def analyze_node(state: AgentState):
    analyzed = []
    for item in state["collected_data"]:
        content = item.get("content", "")
        related = vector_store.similarity_search(content, k=2)
        related_context = "\n".join(d.page_content for d in related)
        response = analysis_llm.invoke([HumanMessage(content=(
            "Analyze the following research content.\n\n"
            f"Content:\n{content}\n\n"
            f"Related prior research from memory:\n{related_context}\n\n"
            "Generate:\n1. Summary\n2. Importance Score (1-10)\n3. Business Impact"
        ))])
        analyzed.append({"source": item.get("url", "Unknown"), "analysis": response.content})
    return {"analyzed_data": analyzed, "execution_logs": log(f"Analyzed {len(analyzed)} sources.")}

def evaluate_node(state: AgentState):
    result = evaluate_quality(state["analyzed_data"])
    return {"quality_score": result.score, "execution_logs": log(f"Quality score = {result.score} ({result.reasoning})")}

def report_node(state: AgentState):
    response = report_llm.invoke([HumanMessage(content=(
        "Generate a professional enterprise research report.\n\n"
        f"Topic:\n{state['topic']}\n\n"
        f"Research Analysis:\n{state['analyzed_data']}\n\n"
        "The report must include:\n- Executive Summary\n- Key Findings\n- Risks\n- Opportunities\n- Strategic Recommendations"
    ))])
    return {"final_report": response.content, "execution_logs": log("Final report generated.")}

def audit_node(state: AgentState):
    return {"execution_logs": log(f"Audit complete. Iterations: {state['iteration_count']}, final quality: {state['quality_score']}.")}

# ============================================================
# STEP 5 — THE CONDITIONAL EDGE
# ============================================================
def quality_router(state: AgentState) -> str:
    score = state["quality_score"]
    iteration = state["iteration_count"]
    if score >= QUALITY_THRESHOLD:
        print(f"Quality {score} >= {QUALITY_THRESHOLD} -> report.")
        return "report"
    if iteration >= MAX_RESEARCH_ITERATIONS:
        print(f"Max iterations ({iteration}) reached -> report anyway.")
        return "report"
    print(f"Quality {score} < {QUALITY_THRESHOLD} -> recollecting.")
    return "collect"

# ============================================================
# STEP 6 — WIRE THE GRAPH
# ============================================================
workflow = StateGraph(AgentState)

workflow.add_node("collect", collect_node)
workflow.add_node("store_memory", store_memory_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("report", report_node)
workflow.add_node("audit", audit_node)

workflow.add_edge(START, "collect")
workflow.add_edge("collect", "store_memory")
workflow.add_edge("store_memory", "analyze")
workflow.add_edge("analyze", "evaluate")

workflow.add_conditional_edges("evaluate", quality_router, {"collect": "collect", "report": "report"})

workflow.add_edge("report", "audit")
workflow.add_edge("audit", END)

# ============================================================
# STEP 7 — COMPILE & RUN
# ============================================================
app = workflow.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    initial_state = {
        "topic": "Enterprise Agentic AI Systems",
        "search_query": "",
        "collected_data": [],
        "analyzed_data": [],
        "quality_score": 0,
        "iteration_count": 0,
        "final_report": "",
        "execution_logs": [],
    }
    
    config = {"configurable": {"thread_id": "lab-day2-run-1"}}
    final_state = None
    for update in app.stream(initial_state, config, stream_mode="values"):
        final_state = update

    print("\n================================================")
    print("FINAL ENTERPRISE RESEARCH REPORT")
    print("================================================")
    print(final_state["final_report"])