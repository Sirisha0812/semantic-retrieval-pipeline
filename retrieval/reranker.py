from sentence_transformers import CrossEncoder

from retrieval.vector_store import Document


class ReRanker:
    def __init__(self):
        # Load the cross-encoder model once during initialization
        # Model trained on MS MARCO dataset for passage reranking
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Warmup: Run one prediction to trigger PyTorch JIT compilation
        # This burns the first-call compilation cost before real usage
        self.model.predict([("warmup query", "warmup doc")])

        # Track how many times rerank() has been called
        self.rerank_count = 0

        # Track total number of query-document pairs scored across all calls
        self.total_pairs_scored = 0

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5
    ) -> list[Document]:
        """
        Rerank documents using cross-encoder for better relevance.

        Args:
            query: The search query
            documents: List of documents from vector search
            top_k: Number of top documents to return

        Returns:
            List of top_k documents sorted by cross-encoder score (highest first)
        """
        # Increment rerank call counter
        self.rerank_count += 1

        # Build query-document pairs for batch scoring
        # Format: [(query, doc1_text), (query, doc2_text), ...]
        pairs = [(query, doc.text) for doc in documents]

        # Track total pairs scored
        self.total_pairs_scored += len(pairs)

        # Score ALL pairs in ONE batch call (10x faster than looping)
        # Returns array of scores, one per pair
        scores = self.model.predict(pairs)

        # Create new Document objects with updated cross-encoder scores
        # Keep same id and text, replace score with cross-encoder score
        reranked_docs = [
            Document(
                id=doc.id,
                text=doc.text,
                score=float(score)
            )
            for doc, score in zip(documents, scores)
        ]

        # Sort by score descending (highest relevance first)
        reranked_docs.sort(key=lambda doc: doc.score, reverse=True)

        # Return only top_k documents
        return reranked_docs[:top_k]
