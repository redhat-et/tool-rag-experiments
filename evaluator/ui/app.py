# Streamlit Demo – Tool RAG Playground
# -------------------------------------------------------------
# This app is a polished demo UI to showcase the Tool RAG solution.
# It presents:
#   1) A chat window to run queries with or without Tool RAG
#   2) A "Tool Definitions" viewer for MCP/LangChain tools
#   3) A sidebar with system settings
# -------------------------------------------------------------
import asyncio
import json
import os
import random
from typing import Any, Dict, List, Tuple

import streamlit as st

from evaluator.algorithms.baseline_algorithm import BaselineAlgorithm
from evaluator.algorithms.tool_rag_algorithm import ToolRagAlgorithm
from evaluator.components.data_provider import QuerySpecification, get_queries, get_tools_from_queries
from evaluator.components.llm_provider import get_llm
from evaluator.components.mcp_proxy import MCPProxyManager
from evaluator.config.defaults import DEFAULT_CONFIG
from evaluator.config.schema import ModelConfig, EnvironmentConfig, DatasetConfig
from evaluator.interfaces.algorithm import AlgoResponse

# Optional LangChain imports (the app will work with mocks if missing)
try:
    from langchain_core.tools import BaseTool, StructuredTool
    from langchain_core.runnables import Runnable
    from langchain_core.language_models import BaseChatModel
except ImportError:
    BaseTool = object  # type: ignore
    StructuredTool = None  # type: ignore
    Runnable = object  # type: ignore
    BaseChatModel = object  # type: ignore

# -------------------------------------------------------------
# UI THEME TWEAKS
# -------------------------------------------------------------
APP_TITLE = "Tool RAG Playground"
APP_SUBTITLE = "A demo UI for the Tool RAG framework"

CUSTOM_CSS = """
<style>
/******* Global look ******/
:root { --radius: 16px; }
.block-container { padding-top: 2rem !important; }

/******* Header badge ******/
.badge {
  display:inline-flex; align-items:center; gap:.4rem;
  padding:.25rem .6rem; border-radius:999px; font-size:.78rem;
  background:linear-gradient(135deg, #eef2ff, #f5f3ff);
  border:1px solid #e5e7eb; color:#374151;
}
.badge .dot { width:.5rem; height:.5rem; border-radius:999px; background:#6366f1; display:inline-block; }

/******* Chat bubbles ******/
.chat-bubble {
  border-radius: var(--radius);
  padding: .8rem 1rem;
  margin:.35rem 0;
}
.user-bubble {
  background:#ecfeff;
  border:1px solid #bae6fd;
  color:#0f172a;           /* NEW: force dark text so it's readable in dark theme */
}
.ai-bubble {
  background:white;
  border:1px solid #e5e7eb;
  box-shadow:0 1px 2px rgba(0,0,0,.04);
  color:#111827;           /* NEW: force dark text on white bubble */
}
.role {
  font-size:.75rem;
  opacity:.7;
  margin-bottom:.25rem;
}

/******* Small cards ******/
.kv-card {
  border:1px solid #e5e7eb;
  border-radius: var(--radius);
  padding:.75rem;
  background:white;
}
.kv-row {
  display:flex;
  justify-content:space-between;
  gap:.75rem;
  font-size:.9rem;
  padding:.2rem 0;
}
.kv-key { opacity:.7; }

/******* Tool list ******/
.tool-card {
  border:1px solid #e5e7eb;
  border-radius: var(--radius);
  padding:1rem;
  background:white;
  margin-bottom:.75rem;
}
.tool-title { font-weight:600; }
.tool-code {
  font-size:.85rem;
  background:#0b1020;
  color:#f0f3ff;
  border-radius:12px;
  padding:.75rem;
  overflow:auto;
}

/******* Subtle separators ******/
hr.soft {
  border:none;
  border-top:1px solid #eee;
  margin: 1rem 0;
}

@media (prefers-color-scheme: dark) {
  .ai-bubble {
    border-color:#4b5563;
    box-shadow:0 1px 2px rgba(0,0,0,.8);
  }
  .user-bubble {
    border-color:#475569;
  }
}
</style>
"""


class MockAlgorithm:
    def __init__(self, settings: Dict[str, Any]):
        self.model = None
        self.tools = None
        self.settings = settings
        self.ready = False

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        # In the mock, we just store references
        self.model = model
        self.tools = tools
        self.ready = True

    async def process_query(self, query: QuerySpecification) -> AlgoResponse:
        assert self.ready, "Engine not set up. Call set_up first."
        tool_names = ", ".join(getattr(t, "name", "tool") for t in (self.tools or [])) or "no tools"
        mode = self.settings.get("mode", "No RAG")
        response = f"[MOCK - {mode}] You asked: '{query.query}'.\nUsing {tool_names}.\nSettings: {json.dumps(self.settings)}"
        return {"response": response}, None


def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("engine", None)
    st.session_state.setdefault("model", None)

    st.session_state.setdefault("mcp_proxy", None)
    st.session_state.setdefault("tools", [])

    st.session_state.setdefault("initialized", False)
    st.session_state.setdefault("current_mode", None)


def demo_tools() -> List[BaseTool]:
    if not st.session_state["mcp_proxy"]:
        environment = EnvironmentConfig(model_id="granite32-8b")
        dataset_config = DatasetConfig.model_validate(DEFAULT_CONFIG["data"])
        dataset_config.queries_num = 60
        queries = get_queries(environment, dataset_config)
        tool_specs = get_tools_from_queries(queries)

        proxy_manager = MCPProxyManager(int(os.getenv("MCP_PROXY_LOCAL_PORT")))
        st.session_state["mcp_proxy"] = proxy_manager.run_mcp_proxy(tool_specs, init_client=True)

    st.session_state["tools"] = list(reversed(asyncio.run(st.session_state["mcp_proxy"].get_tools())))
    random.shuffle(st.session_state["tools"])
    return st.session_state["tools"]


def sidebar() -> Tuple[str, Dict[str, Any], List[BaseTool]]:
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        mode = st.radio(
            "Mode",
            options=["No Tool RAG", "Tool RAG"],
            help="Run plain LLM or your Tool-RAG pipeline",
        )

        st.markdown("#### Model")
        provider = st.selectbox("Provider", ["vLLM", "Mock"], index=0)
        model_url = st.text_input("URL", value=os.getenv("GRANITE_MODEL_URL", ""))
        demo_model_id = "granite32-8b"
        demo_model_config = ModelConfig(id=demo_model_id, url=model_url, provider_id="vllm")

        st.markdown("#### Tool RAG Settings")
        top_k = st.number_input("Top K", min_value=1, max_value=50, value=10)

        engine_settings: Dict[str, Any] = {
            "available_tools_per_query": None,

            "top_k": int(top_k),
            "embedding_model_id": "intfloat/e5-large-v2",
            "max_document_size": 256,
            "cross_encoder_model_name": "BAAI/bge-reranker-large",
        }

        tools: List[BaseTool] = demo_tools()

        st.markdown("---")
        if st.button("Initialize / Re-initialize Engine", use_container_width=True):
            # Build model and engine, stash in state
            if provider == "vLLM":
                model = get_llm(demo_model_id, [demo_model_config])
                engine_cls = BaselineAlgorithm if mode == "No Tool RAG" else ToolRagAlgorithm
                engine = engine_cls(settings=engine_settings, model_config=[demo_model_config])
            else:  # provider == "Mock"
                model = None
                engine = MockAlgorithm(engine_settings)
            try:
                engine.set_up(model=model, tools=tools)

                # Save core runtime objects
                st.session_state["engine"] = engine
                st.session_state["model"] = model
                st.session_state["initialized"] = True

                # Reset chat history
                st.session_state["messages"] = []

                # Record active mode for next time
                st.session_state["current_mode"] = mode

                st.toast("Engine initialized", icon="✅")
            except Exception as e:
                st.session_state["initialized"] = False
                st.error(f"Engine setup failed: {e}")

        return mode, engine_settings, tools


def header():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    left, right = st.columns([0.75, 0.25])
    with left:
        st.title(APP_TITLE)
        st.caption(APP_SUBTITLE)
        st.markdown('<span class="badge"><span class="dot"></span> Live demo UI</span>', unsafe_allow_html=True)
    with right:
        if st.session_state.get("initialized"):
            st.success("Ready")
        else:
            st.info("Not initialized")


def render_tool_viewer(tools: List[BaseTool]):
    st.subheader("🔧 Tool Definitions")

    if tools:
        for t in tools:
            name = getattr(t, "name", "tool")
            desc = getattr(t, "description", "")
            with st.container(border=True):
                st.markdown(f"**{name}**")
                if desc:
                    st.caption(desc)
                # Try to show args / schema
                schema = {}
                try:
                    schema = getattr(t, "args_schema", None)
                    if schema is not None and hasattr(schema, "schema"):
                        schema = schema.schema()
                except Exception:
                    pass
                st.json(schema or {}, expanded=False)
    else:
        st.info("No tools loaded.")


def render_chat(mode):
    st.subheader("💬 Chat")

    # Messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            bubble_cls = "user-bubble" if msg["role"] == "user" else "ai-bubble"
            st.markdown(f"<div class='chat-bubble {bubble_cls}'>"
                        f"<div class='role'>{msg['role'].upper()}</div>"
                        f"{msg['content']}"
                        f"</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask a question…")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                engine = st.session_state.get("engine")
                if engine is None:
                    raise RuntimeError("Engine is not initialized. Use the sidebar to initialize.")

                query_specification = QuerySpecification(id="0", query=prompt, golden_tools={}, demo_mode=True)
                reply = asyncio.run(engine.process_query(query_specification))[0]["messages"][-1].content
                placeholder.markdown(
                    f"<div class='chat-bubble ai-bubble'><div class='role'>ASSISTANT</div>{reply}</div>",
                    unsafe_allow_html=True,
                )
                st.session_state["messages"].append({"role": "assistant", "content": str(reply)})
            except Exception as e:
                placeholder.error(f"Error: {e}")


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧰", layout="wide")
    init_state()
    header()

    # Sidebar
    mode, settings, tools = sidebar()

    # Body tabs
    tab_chat, tab_tools= st.tabs(["💬 Chat", "🔧 Tools"])

    with tab_chat:
        # Quick status card
        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(label="Mode", value=mode)
            with c2:
                st.metric(label="Tools loaded", value=len(tools))
            with c3:
                st.metric(label="Top K", value=settings.get('top_k', '-'))

        st.markdown("<hr class='soft'>", unsafe_allow_html=True)
        render_chat(mode)

    with tab_tools:
        render_tool_viewer(st.session_state.get("tools", []))


if __name__ == "__main__":
    main()
