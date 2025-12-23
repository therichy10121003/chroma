"""
RAG (Retrieval-Augmented Generation) algorithms and strategies.

This module provides various algorithms for combining retrieval results with
generative models to produce better answers.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings


@dataclass
class RAGConfig:
    """Configuration for RAG strategies."""

    top_k: int = 5
    """Number of top documents to retrieve"""

    relevance_threshold: float = 0.0
    """Minimum relevance score for including documents"""

    max_context_length: int = 4000
    """Maximum characters in combined context"""

    rerank: bool = False
    """Whether to rerank results before generation"""

    fusion_strategy: str = "concatenate"
    """Strategy for combining multiple documents: concatenate, summarize, weighted"""

    include_metadata: bool = False
    """Whether to include document metadata in context"""


class RAGAlgorithm:
    """Base class for RAG algorithms."""

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG algorithm.

        Args:
            config: RAG configuration
        """
        self.config = config or RAGConfig()

    def prepare_context(
        self,
        documents: List[str],
        distances: Optional[List[float]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Prepare context documents for generation.

        Args:
            documents: Retrieved documents
            distances: Optional similarity distances
            metadatas: Optional document metadata

        Returns:
            Processed documents ready for generation
        """
        raise NotImplementedError


class SimpleRAG(RAGAlgorithm):
    """
    Simple RAG: Concatenate top-k documents.

    This is the most straightforward RAG approach - just take the top-k
    most relevant documents and pass them as context.
    """

    def prepare_context(
        self,
        documents: List[str],
        distances: Optional[List[float]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Prepare context by taking top-k documents.

        Args:
            documents: Retrieved documents
            distances: Optional similarity distances
            metadatas: Optional document metadata

        Returns:
            Top-k documents
        """
        # Filter by threshold if distances provided
        if distances is not None and self.config.relevance_threshold > 0:
            filtered_docs = [
                doc
                for doc, dist in zip(documents, distances)
                if dist <= self.config.relevance_threshold
            ]
        else:
            filtered_docs = documents

        # Take top-k
        return filtered_docs[: self.config.top_k]


class RerankingRAG(RAGAlgorithm):
    """
    Reranking RAG: Rerank documents using a more sophisticated model.

    Uses cross-encoder or other reranking models to better score
    document relevance to the query.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        reranker: Optional[Callable[[str, List[str]], List[Tuple[str, float]]]] = None,
    ):
        """
        Initialize reranking RAG.

        Args:
            config: RAG configuration
            reranker: Optional custom reranking function
        """
        super().__init__(config)
        self.reranker = reranker

    def prepare_context(
        self,
        documents: List[str],
        distances: Optional[List[float]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None,
    ) -> List[str]:
        """
        Prepare context by reranking documents.

        Args:
            documents: Retrieved documents
            distances: Optional similarity distances
            metadatas: Optional document metadata
            query: Query for reranking (required if reranker is provided)

        Returns:
            Reranked top-k documents
        """
        if self.reranker is not None and query is not None:
            # Use custom reranker
            reranked = self.reranker(query, documents)
            documents = [doc for doc, score in reranked]
        elif distances is not None:
            # Sort by distance (lower is better)
            docs_with_scores = list(zip(documents, distances))
            docs_with_scores.sort(key=lambda x: x[1])
            documents = [doc for doc, _ in docs_with_scores]

        return documents[: self.config.top_k]


class HybridRAG(RAGAlgorithm):
    """
    Hybrid RAG: Combine multiple retrieval strategies.

    Fuses results from different retrieval methods (dense, sparse, keyword)
    using reciprocal rank fusion or other fusion algorithms.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        fusion_weight: float = 0.5,
    ):
        """
        Initialize hybrid RAG.

        Args:
            config: RAG configuration
            fusion_weight: Weight for fusion (0=prefer first, 1=prefer second)
        """
        super().__init__(config)
        self.fusion_weight = fusion_weight

    def reciprocal_rank_fusion(
        self,
        results_list: List[List[Tuple[str, float]]],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion.

        Args:
            results_list: List of result lists, each with (doc, score) tuples
            k: RRF parameter (default 60)

        Returns:
            Fused and ranked results
        """
        # Calculate RRF scores
        doc_scores: Dict[str, float] = {}

        for results in results_list:
            for rank, (doc, _) in enumerate(results):
                if doc not in doc_scores:
                    doc_scores[doc] = 0.0
                doc_scores[doc] += 1.0 / (k + rank + 1)

        # Sort by RRF score
        fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    def prepare_context(
        self,
        documents: List[str],
        distances: Optional[List[float]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        alternative_docs: Optional[List[str]] = None,
        alternative_distances: Optional[List[float]] = None,
    ) -> List[str]:
        """
        Prepare context by fusing multiple retrieval results.

        Args:
            documents: Primary retrieved documents
            distances: Primary similarity distances
            metadatas: Optional document metadata
            alternative_docs: Alternative retrieval results
            alternative_distances: Alternative distances

        Returns:
            Fused top-k documents
        """
        if alternative_docs is None:
            # Fall back to simple RAG
            return SimpleRAG(self.config).prepare_context(documents, distances, metadatas)

        # Prepare results for fusion
        primary_results = [
            (doc, dist)
            for doc, dist in zip(
                documents, distances or [0.0] * len(documents)
            )
        ]
        alternative_results = [
            (doc, dist)
            for doc, dist in zip(
                alternative_docs,
                alternative_distances or [0.0] * len(alternative_docs),
            )
        ]

        # Apply reciprocal rank fusion
        fused = self.reciprocal_rank_fusion([primary_results, alternative_results])

        # Return top-k
        return [doc for doc, _ in fused[: self.config.top_k]]


class ContextCompressor:
    """
    Compress context to fit within token limits.

    Uses various strategies to reduce context size while preserving
    the most relevant information.
    """

    def __init__(self, max_length: int = 4000):
        """
        Initialize context compressor.

        Args:
            max_length: Maximum characters in compressed context
        """
        self.max_length = max_length

    def compress(
        self,
        documents: List[str],
        strategy: str = "truncate",
    ) -> List[str]:
        """
        Compress documents to fit within length limit.

        Args:
            documents: Documents to compress
            strategy: Compression strategy (truncate, summarize, extract)

        Returns:
            Compressed documents
        """
        if strategy == "truncate":
            return self._truncate(documents)
        elif strategy == "extract":
            return self._extract_sentences(documents)
        else:
            warnings.warn(f"Unknown compression strategy: {strategy}, using truncate")
            return self._truncate(documents)

    def _truncate(self, documents: List[str]) -> List[str]:
        """Truncate documents to fit within max_length."""
        total_length = sum(len(doc) for doc in documents)

        if total_length <= self.max_length:
            return documents

        # Proportionally truncate each document
        ratio = self.max_length / total_length
        return [doc[: int(len(doc) * ratio)] for doc in documents]

    def _extract_sentences(self, documents: List[str]) -> List[str]:
        """Extract most relevant sentences from documents."""
        # Simple sentence extraction (could be improved with ML)
        sentences = []
        for doc in documents:
            doc_sentences = [s.strip() for s in doc.split(".") if s.strip()]
            sentences.extend(doc_sentences[:2])  # Take first 2 sentences per doc

        # Combine until max_length
        compressed = []
        current_length = 0
        for sent in sentences:
            if current_length + len(sent) + 1 <= self.max_length:
                compressed.append(sent)
                current_length += len(sent) + 1
            else:
                break

        return [". ".join(compressed) + "."]


class AdaptiveRAG(RAGAlgorithm):
    """
    Adaptive RAG: Dynamically adjust retrieval based on query complexity.

    Analyzes the query to determine the best retrieval strategy,
    number of documents, and generation parameters.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize adaptive RAG."""
        super().__init__(config)
        self.compressor = ContextCompressor(config.max_context_length)

    def analyze_query_complexity(self, query: str) -> str:
        """
        Analyze query to determine complexity.

        Args:
            query: User query

        Returns:
            Complexity level: simple, medium, complex
        """
        # Simple heuristic based on query length and structure
        words = query.split()
        if len(words) <= 5:
            return "simple"
        elif len(words) <= 15:
            return "medium"
        else:
            return "complex"

    def prepare_context(
        self,
        documents: List[str],
        distances: Optional[List[float]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None,
    ) -> List[str]:
        """
        Adaptively prepare context based on query.

        Args:
            documents: Retrieved documents
            distances: Optional similarity distances
            metadatas: Optional document metadata
            query: Query for analysis

        Returns:
            Adaptively prepared context
        """
        if query is not None:
            complexity = self.analyze_query_complexity(query)

            # Adjust top_k based on complexity
            if complexity == "simple":
                top_k = min(3, self.config.top_k)
            elif complexity == "medium":
                top_k = self.config.top_k
            else:
                top_k = min(self.config.top_k + 2, len(documents))
        else:
            top_k = self.config.top_k

        # Get top documents
        selected_docs = documents[:top_k]

        # Compress if needed
        return self.compressor.compress(selected_docs)


# Factory function for creating RAG algorithms
def create_rag_algorithm(
    strategy: str = "simple",
    config: Optional[RAGConfig] = None,
    **kwargs: Any,
) -> RAGAlgorithm:
    """
    Create a RAG algorithm instance.

    Args:
        strategy: RAG strategy (simple, reranking, hybrid, adaptive)
        config: RAG configuration
        **kwargs: Additional strategy-specific parameters

    Returns:
        RAG algorithm instance
    """
    strategies = {
        "simple": SimpleRAG,
        "reranking": RerankingRAG,
        "hybrid": HybridRAG,
        "adaptive": AdaptiveRAG,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown RAG strategy: {strategy}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy](config, **kwargs)
