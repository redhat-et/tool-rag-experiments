"""
Base class for tool retrieval algorithms with shared functionality.
"""
import json
import os
import re
import unicodedata
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from langchain.docstore.document import Document
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction

from evaluator.config.defaults import VERBOSE
from evaluator.config.schema import ModelConfig
from evaluator.interfaces.algorithm import Algorithm
from evaluator.utils.utils import log_verbose

if not VERBOSE:
    # Silence gRPC C-core and tracing
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = ""
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

_CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
_WS_RE = re.compile(r'\s+')


class L2Wrapper:
    """Wrapper to L2-normalize embeddings for stable COSINE similarity."""
    def __init__(self, base):
        self.base = base

    def embed_documents(self, texts):
        x = np.array(self.base.embed_documents(texts), dtype=np.float32)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        return x.tolist()

    def embed_query(self, text):
        x = np.array(self.base.embed_query(text), dtype=np.float32)
        x /= (np.linalg.norm(x) + 1e-12)
        return x.tolist()


class BaseRetrievalAlgorithm(Algorithm):
    """
    Base class for algorithms that use vector-based tool retrieval.
    
    Provides shared functionality for:
    - Text preprocessing (unicode normalization, lowercasing, etc.)
    - Tool text composition from tool definitions
    - Vector store initialization with Milvus
    - Hybrid search (dense + sparse/BM25)
    - Reranking with cross-encoders
    - NMS-based diversity filtering
    - Similarity thresholding
    """

    vector_store: Milvus or None
    reranker: CrossEncoderReranker or None
    embeddings: HuggingFaceEmbeddings or None
    tool_name_to_base_tool: Dict[str, BaseTool] or None

    def __init__(self, settings: Dict, model_config: List[ModelConfig], label: str = None):
        super().__init__(settings, model_config, label)
        self.tool_name_to_base_tool = None
        self.vector_store = None
        self.reranker = None
        self.embeddings = None

    def _normalize_similarity_metric(self) -> str:
        """Ensure similarity metric is stored in normalized (upper-case) form."""
        metric = str(self._settings.get("similarity_metric", "COSINE")).upper()
        self._settings["similarity_metric"] = metric
        return metric

    def get_default_settings(self) -> Dict[str, Any]:
        """
        Default settings shared across retrieval-based algorithms.
        Subclasses can override to add or modify settings.
        """
        return {
            # Basic
            "embedding_model_id": "all-MiniLM-L6-v2",
            "similarity_metric": "COSINE",
            "index_type": "FLAT",
            "indexed_tool_def_parts": ["name", "description"],

            # Preprocessing
            "text_preprocessing_operations": None,
            "max_document_size": None,

            # Hybrid search
            "hybrid_mode": False,
            "analyzer_params": None,
            "fusion_type": "rrf",
            "fusion_k": 100,
            "fusion_alpha": 0.5,

            # Reranking
            "cross_encoder_model_name": None,
            "reranker_pool_size": 50,

            # Post-retrieval filtering
            "tau": None,
            "sim_threshold": None,
        }

    # ========== Text Preprocessing ==========

    def _preprocess_text(self, text: str) -> str:
        """Apply configured text preprocessing operations."""
        ops = self._settings.get("text_preprocessing_operations")
        if not ops:
            return text

        t = text
        if "unicode_normalization" in ops or "all" in ops:
            t = unicodedata.normalize('NFC', t)
        if "lowercase" in ops or "all" in ops:
            t = t.lower()
        if "collapse_whitespaces" in ops or "all" in ops:
            t = _WS_RE.sub(' ', t).strip()
        if "split_camel_snake_case" in ops or "all" in ops:
            t = t.replace('_', ' ')  # snake_case -> snake case
            t = _CAMEL_RE.sub(' ', t)  # camelCase -> camel Case

        return t

    def _truncate(self, s: str, sep: str = " ... ") -> str:
        """Truncate text to max_document_size if configured."""
        max_chars = self._settings.get("max_document_size")
        if max_chars is None or len(s) <= max_chars:
            return s

        # Performs a simple head-tail truncation to max_chars characters.
        half = (max_chars - len(sep)) // 2
        return s[:half] + sep + s[-half:]

    # ========== Tool Text Composition ==========

    @staticmethod
    def _render_args_schema(schema: Any) -> str:
        """Render tool arguments schema as concise string."""
        if schema is None:
            return ""

        if isinstance(schema, dict):
            schema_dict = schema
        elif isinstance(schema, str):
            try:
                decoded = json.loads(schema)
            except json.JSONDecodeError:
                decoded = None
            schema_dict = decoded if isinstance(decoded, dict) else {}
        else:
            schema_dict = {}

            def _merge(candidate: Any) -> bool:
                if candidate is None:
                    return False
                for attr in ("model_json_schema", "schema", "json_schema", "dict"):
                    fn = getattr(candidate, attr, None)
                    if callable(fn):
                        try:
                            data = fn()
                        except TypeError:
                            continue
                        if isinstance(data, dict):
                            schema_dict.update(data)
                            return True
                        if isinstance(data, str):
                            try:
                                decoded_inner = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(decoded_inner, dict):
                                schema_dict.update(decoded_inner)
                                return True
                return False

            candidates = [schema]
            if not isinstance(schema, type):
                candidates.append(getattr(schema, "__class__", None))
            for candidate in candidates:
                if _merge(candidate):
                    break

        if not schema_dict:
            return ""

        props = schema_dict.get("properties") or {}
        required = schema_dict.get("required") or []
        ordered = list(required) + [p for p in props.keys() if p not in required]
        # Keep it terse: "name:type" when available
        parts = []
        for name in ordered:
            spec = props.get(name) or {}
            t = spec.get("type")
            if isinstance(t, str):
                parts.append(f"{name}:{t}")
            else:
                parts.append(name)
        return " ".join(parts)

    def _compose_tool_text(self, tool: BaseTool) -> str:
        """Compose indexed text from tool based on configured parts."""
        parts_to_include = self._settings.get("indexed_tool_def_parts", ["name", "description"])
        if not parts_to_include:
            raise ValueError("indexed_tool_def_parts must be a non-empty list")

        segments = []
        for p in parts_to_include:
            if p.lower() == "name":
                val = tool.name
                if val:
                    segments.append(f"name: {val}")
            elif p.lower() == "description":
                val = tool.description
                if val:
                    segments.append(f"desc: {val}")
            elif p.lower() == "args":
                val = self._render_args_schema(tool.args_schema)
                if val:
                    segments.append(f"args: {val}")
            elif p.lower() == "tags":
                tags = tool.tags or []
                if tags:
                    segments.append(f"tags: {' '.join(tags)}")

        if not segments:
            raise ValueError(f"The following tool contains none of the fields listed in indexed_tool_def_parts:\n{tool}")
        
        text = " | ".join(segments)
        # One-pass preprocess + truncation
        text = self._preprocess_text(text)
        text = self._truncate(text)
        return text

    def _create_docs_from_tools(self, tools: List[BaseTool]) -> List[Document]:
        """Create Langchain Documents from tools for indexing."""
        documents = []
        for tool in tools:
            page_content = self._compose_tool_text(tool)
            documents.append(Document(page_content=page_content, metadata={"name": tool.name}))
        return documents

    # ========== Tool mapping & retrieval helpers ==========

    def _update_tool_mapping(self, tools: List[BaseTool]) -> None:
        """Cache tools by name for quick lookup in subclasses."""
        self.tool_name_to_base_tool = {tool.name: tool for tool in tools}

    def _vector_search_with_scores(
        self,
        query_text: str,
        k: int,
    ) -> List[Tuple[Document, float]]:
        """
        Run a similarity search against the vector store, applying hybrid parameters
        and filtering out documents whose tool names are unknown.
        """
        if not self.vector_store or k <= 0:
            return []

        fusion_type, fusion_params = self._get_hybrid_fusion_params()
        search_kwargs: Dict[str, Any] = {"k": k}
        if fusion_type:
            search_kwargs["ranker_type"] = fusion_type
            search_kwargs["ranker_params"] = fusion_params

        results = self.vector_store.similarity_search_with_score(
            query_text,
            **search_kwargs,
        )

        tool_map = self.tool_name_to_base_tool or {}
        filtered: List[Tuple[Document, float]] = []
        for doc, score in results:
            name = doc.metadata.get("name") if doc.metadata else None
            if name and name in tool_map:
                filtered.append((doc, score))
        return filtered

    def _docs_to_tools(self, docs: List[Document], limit: Optional[int] = None) -> List[BaseTool]:
        """
        Convert Documents to BaseTool objects, filtering out missing tools.
        
        Args:
            docs: List of documents with tool metadata
            limit: Optional limit on number of tools to return
            
        Returns:
            List of BaseTool objects, maintaining order from docs
        """
        tools = []
        for doc in docs:
            name = doc.metadata.get("name") if doc.metadata else None
            tool = self.tool_name_to_base_tool.get(name) if name else None
            if tool is not None:
                tools.append(tool)
                if limit is not None and len(tools) >= limit:
                    break
        return tools

    def _get_retrieval_k(self, requested_k: int) -> int:
        """
        Calculate k for vector search, accounting for reranking pool size.
        
        If reranking is enabled, we need to retrieve more candidates than
        the final requested number to ensure quality after reranking.
        
        Args:
            requested_k: The desired number of final results
            
        Returns:
            The number of results to retrieve from vector store
        """
        if self.reranker is not None:
            pool_size = self._settings.get("reranker_pool_size", 50)
            return max(requested_k, pool_size)
        return requested_k

    def _substring_search_tools(
        self,
        query: str,
        limit: int,
        name_weight: int = 2,
        desc_weight: int = 1
    ) -> List[BaseTool]:
        """
        Fallback substring search when vector search fails.
        
        Returns tools ranked by substring match in name/description.
        If no matches found, returns first `limit` tools.
        
        Args:
            query: Search query
            limit: Maximum number of tools to return
            name_weight: Score weight for name matches
            desc_weight: Score weight for description matches
            
        Returns:
            List of matched tools, ranked by relevance
        """
        if not self.tool_name_to_base_tool:
            return []

        all_tools = list(self.tool_name_to_base_tool.values())
        q = query.lower()
        ranked = []

        for tool in all_tools:
            name = (tool.name or "").lower()
            desc = (getattr(tool, "description", "") or "").lower()
            score = (name_weight if q in name else 0) + (desc_weight if q in desc else 0)
            if score > 0:
                ranked.append((score, tool))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in ranked[:limit]] or all_tools[:limit]

    # ========== Vector Store Setup ==========

    def _initialize_embeddings(self) -> Any:
        """Initialize embedding model with optional L2 normalization."""
        metric = self._normalize_similarity_metric()
        self.embeddings = HuggingFaceEmbeddings(model_name=self._settings["embedding_model_id"])
        if metric == "COSINE":
            # L2-normalizing embedding vectors before cosine similarity makes the results more stable
            return L2Wrapper(self.embeddings)
        return self.embeddings

    def _get_index_and_search_params(self) -> Tuple[Any, Any]:
        """Get index and search parameters for Milvus."""
        index_params = {
            "index_type": self._settings["index_type"],
            "metric_type": self._settings["similarity_metric"],
        }
        search_params = {
            "metric_type": self._settings["similarity_metric"],
        }
        
        if self._settings["hybrid_mode"]:
            # Hybrid search: extend params for sparse search
            index_params = [index_params, None]
            search_params = [search_params, {}]
        
        return index_params, search_params

    def _build_vector_store(self, tools: List[BaseTool], collection_name: str, drop_old: bool = True) -> None:
        """Build Milvus vector store for tool retrieval."""
        embeddings = self._initialize_embeddings()
        milvus_uri = os.getenv("MILVUS_URL")
        index_params, search_params = self._get_index_and_search_params()

        log_verbose(f"Building Milvus collection: {collection_name} (drop_old={drop_old})")

        if self._settings["hybrid_mode"]:
            # Hybrid search: dense + sparse (BM25)
            log_verbose("Enabling hybrid search (dense + BM25)")
            self.vector_store = Milvus.from_documents(
                documents=self._create_docs_from_tools(tools),
                embedding=embeddings,
                collection_name=collection_name,
                connection_args={"uri": milvus_uri},
                drop_old=drop_old,
                index_params=index_params,
                search_params=search_params,
                builtin_function=BM25BuiltInFunction(analyzer_params=self._settings["analyzer_params"]),
                vector_field=["dense", "sparse"],
            )
        else:
            # Dense-only search
            self.vector_store = Milvus.from_documents(
                documents=self._create_docs_from_tools(tools),
                embedding=embeddings,
                collection_name=collection_name,
                connection_args={"uri": milvus_uri},
                drop_old=drop_old,
                index_params=index_params,
                search_params=search_params,
            )

    # ========== Reranking Setup ==========

    def _initialize_reranker(self, top_k: int) -> None:
        """Initialize cross-encoder reranker if configured."""
        if self._settings.get("cross_encoder_model_name"):
            log_verbose(f"Initializing reranker: {self._settings['cross_encoder_model_name']}")
            self.reranker = CrossEncoderReranker(
                model=HuggingFaceCrossEncoder(model_name=self._settings["cross_encoder_model_name"]),
                top_n=top_k,
            )

    # ========== Retrieval and Post-processing ==========

    def _get_hybrid_fusion_params(self) -> Tuple[str, Dict]:
        """Get fusion type and params for hybrid search."""
        if not self._settings["hybrid_mode"]:
            return None, None

        fusion_type = self._settings["fusion_type"]
        if fusion_type == "weighted":
            alpha = self._settings["fusion_alpha"]
            fusion_params = {"weights": [alpha, 1 - alpha]}
        elif fusion_type == "rrf":
            fusion_params = {"k": self._settings["fusion_k"]}
        else:
            raise ValueError(f"Unsupported hybrid fusion type: {fusion_type}")
        
        return fusion_type, fusion_params

    def _threshold_results(self, docs_and_scores: List[Tuple[Document, float]]) -> List[Document]:
        """Filter search results by tau threshold."""
        tau = self._settings.get("tau")
        if tau is None:
            return [d for (d, s) in docs_and_scores]

        metric = self._settings["similarity_metric"]

        if metric.upper() in {"COSINE", "IP"}:
            kept = [(d, s) for (d, s) in docs_and_scores if s >= tau]
            kept.sort(key=lambda x: x[1], reverse=True)  # higher is better
        elif metric.upper() == "L2":
            kept = [(d, s) for (d, s) in docs_and_scores if s <= tau]
            kept.sort(key=lambda x: x[1])  # lower is better
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return [d for (d, s) in kept]

    def _embed_docs(self, docs: List[Document]) -> np.ndarray:
        """Encode docs -> (n, d), then L2-normalize for cosine similarity."""
        texts = [d.page_content for d in docs]
        vecs = np.asarray(self.embeddings.embed_documents(texts), dtype=np.float32)
        # L2-normalize the vectors
        n = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(n, 1e-12)

    def _diversify_nms(self, docs: List[Document]) -> List[Document]:
        """Apply NMS-based diversity filtering to remove highly similar tools."""
        if not docs:
            return []

        embedded = self._embed_docs(docs)  # (n,d), L2-normalized
        kept_indices = []
        for i in range(len(docs)):
            v = embedded[i]
            # if nothing kept yet, keep the first one (best by reranker)
            if not kept_indices:
                kept_indices.append(i)
                continue
            # compute similarity to all kept
            sims = (embedded[kept_indices] @ v)  # (len(kept),)
            if float(np.max(sims)) < self._settings["sim_threshold"]:
                kept_indices.append(i)

        return [docs[i] for i in kept_indices]

    def _postprocess_results(
        self,
        docs_and_scores: List[Tuple[Document, float]],
        query: str,
        limit: Optional[int] = None,
    ) -> List[Document]:
        """
        Optionally filters and/or reranks results from the vector DB search.
        """
        if self._settings.get("tau") is not None:
            # tau is specified - thresholding is enabled
            docs = self._threshold_results(docs_and_scores)
        else:
            docs = [d for (d, s) in docs_and_scores]

        if self._settings.get("cross_encoder_model_name") is not None and self.reranker:
            # reranking is enabled
            if limit is not None and hasattr(self.reranker, "top_n"):
                current_top_n = getattr(self.reranker, "top_n", 0) or 0
                if limit > current_top_n:
                    self.reranker.top_n = limit  # type: ignore[attr-defined]
            docs = self.reranker.compress_documents(docs, query)

        if self._settings.get("sim_threshold") is not None:
            # NMS-based diversity filtering is enabled
            docs = self._diversify_nms(docs)

        return docs

    # ========== Shared parsing utilities ==========

    @staticmethod
    def _safe_json_parse(text: str):
        """Safely parse JSON from potentially malformed text."""
        s = (text or "").strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            m = re.search(r"(\{.*}|\[.*])", s, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _lines(text: str) -> List[str]:
        """Extract non-empty lines from text."""
        return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

    @staticmethod
    def _dedup_keep_order(xs: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    @staticmethod
    def _strip_numbering(s: str) -> str:
        """Strip leading numbering/bullets from a line."""
        return re.sub(r"^\s*(?:[-*]|\d+[).:]?)\s*", "", s).strip().rstrip(".")

