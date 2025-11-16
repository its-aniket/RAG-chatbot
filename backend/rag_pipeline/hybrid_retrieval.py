"""
Hybrid Retrieval System for RAG
Combines dense vector search with sparse keyword search for better retrieval
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
from dataclasses import dataclass
from collections import Counter
import math

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Enhanced result structure for hybrid retrieval"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    dense_score: float
    sparse_score: float
    hybrid_score: float
    rank_fusion_score: float

class HybridRetriever:
    def __init__(
        self,
        vector_store,  # ChromaDB VectorStore
        embedder,     # EmbeddingGenerator
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        max_docs_for_sparse: int = 1000,
        enable_query_expansion: bool = True
    ):
        """
        Initialize hybrid retrieval system
        
        Args:
            vector_store: ChromaDB vector store instance
            embedder: Embedding generator instance
            dense_weight: Weight for dense (vector) search
            sparse_weight: Weight for sparse (keyword) search  
            max_docs_for_sparse: Maximum documents to consider for sparse search
            enable_query_expansion: Whether to expand queries
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.max_docs_for_sparse = max_docs_for_sparse
        self.enable_query_expansion = enable_query_expansion
        
        # Initialize TF-IDF vectorizer for sparse retrieval
        self.tfidf_vectorizer = None
        self.document_texts = {}
        self.document_metadata = {}
        
        logger.info("Hybrid retrieval system initialized")
    
    def index_documents_for_sparse_search(
        self, 
        filter_criteria: Optional[Dict[str, Any]] = None,
        document_ids: Optional[Union[str, List[str]]] = None
    ):
        """
        Index documents for sparse (keyword) search
        
        Args:
            filter_criteria: Optional metadata filters
            document_ids: Optional document ID filters
        """
        try:
            # Get documents from vector store
            query_params = {
                "limit": self.max_docs_for_sparse,
                "include": ["documents", "metadatas"]
            }
            
            # Build filters
            final_filter = self._build_filter(filter_criteria, document_ids)
            if final_filter:
                query_params["where"] = final_filter
            
            results = self.vector_store.collection.get(**query_params)
            
            if not results["ids"]:
                logger.warning("No documents found for sparse indexing")
                return
            
            # Prepare documents for TF-IDF
            documents = []
            self.document_texts = {}
            self.document_metadata = {}
            
            for i, doc_id in enumerate(results["ids"]):
                text = results["documents"][i]
                metadata = results["metadatas"][i]
                
                # Clean text for better keyword matching
                cleaned_text = self._clean_text_for_sparse(text)
                documents.append(cleaned_text)
                
                self.document_texts[doc_id] = text
                self.document_metadata[doc_id] = metadata
            
            # Create TF-IDF index
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                lowercase=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            logger.info(f"Indexed {len(documents)} documents for sparse search")
            
        except Exception as e:
            logger.error(f"Failed to index documents for sparse search: {e}")
            self.tfidf_vectorizer = None
    
    def _build_filter(
        self, 
        filter_criteria: Optional[Dict[str, Any]], 
        document_ids: Optional[Union[str, List[str]]]
    ) -> Optional[Dict[str, Any]]:
        """Build ChromaDB filter from criteria"""
        final_filter = None
        
        # Handle document ID filtering
        if document_ids:
            if isinstance(document_ids, str):
                doc_filter = {"document_id": document_ids}
            elif isinstance(document_ids, list) and len(document_ids) > 0:
                doc_filter = {"document_id": {"$in": document_ids}}
            else:
                doc_filter = None
            
            # Combine with other filters
            if filter_criteria and doc_filter:
                final_filter = {"$and": [filter_criteria, doc_filter]}
            elif filter_criteria:
                final_filter = filter_criteria
            elif doc_filter:
                final_filter = doc_filter
        else:
            final_filter = filter_criteria
        
        return final_filter
    
    def _clean_text_for_sparse(self, text: str) -> str:
        """Clean text for better sparse (keyword) matching"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        
        # Remove numbers that don't provide semantic value
        text = re.sub(r'\b\d{4,}\b', '', text)  # Remove long numbers
        
        return text.strip().lower()
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms
        Simple implementation - can be enhanced with word embeddings
        """
        if not self.enable_query_expansion:
            return query
        
        # Simple expansion rules (can be enhanced)
        expansion_map = {
            'ai': ['artificial intelligence', 'machine learning', 'neural networks'],
            'ml': ['machine learning', 'artificial intelligence', 'algorithms'],
            'deep learning': ['neural networks', 'artificial intelligence'],
            'computer': ['computing', 'computational', 'digital'],
            'algorithm': ['method', 'approach', 'technique'],
            'data': ['information', 'dataset', 'statistics']
        }
        
        expanded_terms = []
        query_lower = query.lower()
        
        for term, synonyms in expansion_map.items():
            if term in query_lower:
                expanded_terms.extend(synonyms[:2])  # Add top 2 synonyms
        
        if expanded_terms:
            expanded_query = query + ' ' + ' '.join(expanded_terms)
            logger.debug(f"Expanded query: '{query}' -> '{expanded_query}'")
            return expanded_query
        
        return query
    
    def dense_search(
        self,
        query: str,
        top_k: int = 20,
        filter_criteria: Optional[Dict[str, Any]] = None,
        document_ids: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Perform dense (vector) search"""
        try:
            # Expand query if enabled
            expanded_query = self.expand_query(query)
            
            results = self.vector_store.search_by_query(
                query=expanded_query,
                embedder=self.embedder,
                top_k=top_k,
                filter_criteria=filter_criteria,
                document_ids=document_ids
            )
            
            logger.debug(f"Dense search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def sparse_search(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Perform sparse (keyword/TF-IDF) search"""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            logger.warning("TF-IDF index not available for sparse search")
            return []
        
        try:
            # Expand and clean query
            expanded_query = self.expand_query(query)
            cleaned_query = self._clean_text_for_sparse(expanded_query)
            
            # Vectorize query
            query_vector = self.tfidf_vectorizer.transform([cleaned_query])
            
            # Compute similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            doc_ids = list(self.document_texts.keys())
            
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include non-zero similarities
                    doc_id = doc_ids[idx]
                    results.append({
                        "chunk_id": doc_id,
                        "text": self.document_texts[doc_id],
                        "metadata": self.document_metadata[doc_id],
                        "similarity_score": float(similarities[idx]),
                        "search_type": "sparse"
                    })
            
            logger.debug(f"Sparse search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF)
        
        Args:
            dense_results: Results from vector search
            sparse_results: Results from keyword search
            k: RRF parameter (higher values = less weight to rank position)
        """
        # Create lookup for results
        all_docs = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            doc_id = result["chunk_id"]
            all_docs[doc_id] = {
                "chunk_id": doc_id,
                "text": result["text"],
                "metadata": result["metadata"],
                "dense_score": result["similarity_score"],
                "dense_rank": rank + 1,
                "sparse_score": 0.0,
                "sparse_rank": float('inf')
            }
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = result["chunk_id"]
            if doc_id in all_docs:
                all_docs[doc_id]["sparse_score"] = result["similarity_score"]
                all_docs[doc_id]["sparse_rank"] = rank + 1
            else:
                all_docs[doc_id] = {
                    "chunk_id": doc_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "dense_score": 0.0,
                    "dense_rank": float('inf'),
                    "sparse_score": result["similarity_score"],
                    "sparse_rank": rank + 1
                }
        
        # Calculate fusion scores
        fused_results = []
        
        for doc_id, doc_data in all_docs.items():
            # RRF score
            dense_rrf = 1 / (k + doc_data["dense_rank"])
            sparse_rrf = 1 / (k + doc_data["sparse_rank"])
            rrf_score = dense_rrf + sparse_rrf
            
            # Weighted hybrid score
            hybrid_score = (
                self.dense_weight * doc_data["dense_score"] + 
                self.sparse_weight * doc_data["sparse_score"]
            )
            
            result = RetrievalResult(
                chunk_id=doc_id,
                text=doc_data["text"],
                metadata=doc_data["metadata"],
                dense_score=doc_data["dense_score"],
                sparse_score=doc_data["sparse_score"],
                hybrid_score=hybrid_score,
                rank_fusion_score=rrf_score
            )
            
            fused_results.append(result)
        
        # Sort by RRF score (primary) and hybrid score (secondary)
        fused_results.sort(
            key=lambda x: (x.rank_fusion_score, x.hybrid_score), 
            reverse=True
        )
        
        return fused_results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None,
        document_ids: Optional[Union[str, List[str]]] = None,
        dense_top_k: int = 20,
        sparse_top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse retrieval
        
        Args:
            query: Search query
            top_k: Final number of results to return
            filter_criteria: Optional metadata filters
            document_ids: Optional document ID filters
            dense_top_k: Number of results from dense search
            sparse_top_k: Number of results from sparse search
        """
        try:
            logger.info(f"Starting hybrid search for query: '{query[:50]}...'")
            
            # Index documents for sparse search if needed
            if not self.tfidf_vectorizer:
                logger.info("Indexing documents for sparse search...")
                self.index_documents_for_sparse_search(filter_criteria, document_ids)
            
            # Perform both searches
            dense_results = self.dense_search(
                query, dense_top_k, filter_criteria, document_ids
            )
            sparse_results = self.sparse_search(query, sparse_top_k)
            
            # Filter sparse results by document_ids if specified
            if document_ids:
                allowed_doc_ids = set()
                if isinstance(document_ids, str):
                    allowed_doc_ids.add(document_ids)
                elif isinstance(document_ids, list):
                    allowed_doc_ids.update(document_ids)
                
                sparse_results = [
                    r for r in sparse_results 
                    if r["metadata"].get("document_id") in allowed_doc_ids
                ]
            
            logger.info(f"Dense search: {len(dense_results)} results, Sparse search: {len(sparse_results)} results")
            
            # Combine using reciprocal rank fusion
            fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)
            
            # Convert back to standard format
            final_results = []
            for result in fused_results[:top_k]:
                final_results.append({
                    "chunk_id": result.chunk_id,
                    "text": result.text,
                    "metadata": result.metadata,
                    "similarity_score": result.hybrid_score,
                    "dense_score": result.dense_score,
                    "sparse_score": result.sparse_score,
                    "rank_fusion_score": result.rank_fusion_score,
                    "search_type": "hybrid"
                })
            
            logger.info(f"Hybrid search returned {len(final_results)} final results")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to dense search only
            return self.dense_search(query, top_k, filter_criteria, document_ids)
    
    def get_search_explanation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate explanation of search results for debugging
        """
        if not results:
            return {"message": "No results found"}
        
        explanation = {
            "total_results": len(results),
            "search_type": results[0].get("search_type", "unknown"),
            "score_distribution": {
                "avg_similarity": np.mean([r.get("similarity_score", 0) for r in results]),
                "max_similarity": max([r.get("similarity_score", 0) for r in results]),
                "min_similarity": min([r.get("similarity_score", 0) for r in results])
            }
        }
        
        # Add hybrid-specific stats if available
        if "dense_score" in results[0]:
            explanation["hybrid_stats"] = {
                "avg_dense_score": np.mean([r.get("dense_score", 0) for r in results]),
                "avg_sparse_score": np.mean([r.get("sparse_score", 0) for r in results]),
                "dense_weight": self.dense_weight,
                "sparse_weight": self.sparse_weight
            }
        
        return explanation


# Test function
def test_hybrid_retrieval():
    """Test hybrid retrieval system"""
    from storage import VectorStore
    from embeddings import EmbeddingGenerator
    
    # Initialize components
    embedder = EmbeddingGenerator()
    vector_store = VectorStore()
    hybrid_retriever = HybridRetriever(vector_store, embedder)
    
    # Test queries
    test_queries = [
        "artificial intelligence and machine learning",
        "deep learning neural networks",
        "computer algorithms and data structures"
    ]
    
    print("=== Hybrid Retrieval Test ===")
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        
        # Compare different search methods
        dense_results = hybrid_retriever.dense_search(query, top_k=5)
        sparse_results = hybrid_retriever.sparse_search(query, top_k=5)
        hybrid_results = hybrid_retriever.hybrid_search(query, top_k=5)
        
        print(f"Dense search: {len(dense_results)} results")
        print(f"Sparse search: {len(sparse_results)} results") 
        print(f"Hybrid search: {len(hybrid_results)} results")
        
        if hybrid_results:
            print("\nTop 3 hybrid results:")
            for i, result in enumerate(hybrid_results[:3]):
                print(f"  {i+1}. Hybrid Score: {result['similarity_score']:.4f}")
                print(f"     Dense: {result.get('dense_score', 0):.4f}, Sparse: {result.get('sparse_score', 0):.4f}")
                print(f"     Preview: {result['text'][:80]}...")
        
        # Get explanation
        explanation = hybrid_retriever.get_search_explanation(hybrid_results)
        print(f"\nSearch explanation: {explanation}")
    
    print("\n=== Hybrid Retrieval Test Complete ===")

if __name__ == "__main__":
    test_hybrid_retrieval()