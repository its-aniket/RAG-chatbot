# Multiple Document ID Search Examples

## Enhanced ChromaDB Storage with Multiple Document ID Support

The `VectorStore` class now supports searching across multiple documents with the enhanced `search_similar_chunks` and `search_by_query` functions.

## Usage Examples

```python
from rag_pipeline.storage import VectorStore
from rag_pipeline.embeddings import EmbeddingGenerator

# Initialize components
vector_store = VectorStore(persist_directory="./chroma_db")
embedder = EmbeddingGenerator()

# 1. Search across ALL documents (no filtering)
all_results = vector_store.search_by_query(
    query="machine learning algorithms",
    embedder=embedder,
    top_k=10
)

# 2. Search in a SINGLE document
single_doc_results = vector_store.search_by_query(
    query="neural networks",
    embedder=embedder,
    top_k=5,
    document_ids="document_123"  # Single string
)

# 3. Search across MULTIPLE specific documents
multi_doc_results = vector_store.search_by_query(
    query="artificial intelligence",
    embedder=embedder,
    top_k=8,
    document_ids=["doc_1", "doc_2", "doc_3"]  # List of strings
)

# 4. Combine with other metadata filters
filtered_results = vector_store.search_by_query(
    query="deep learning",
    embedder=embedder,
    top_k=5,
    filter_criteria={"file_type": "pdf"},  # Additional filter
    document_ids=["research_paper_1", "research_paper_2"]
)

# 5. Direct embedding search with multiple docs
import numpy as np

# Your query embedding (384-dim for E5-small)
query_embedding = embedder.embed_query("transformer architecture")

direct_results = vector_store.search_similar_chunks(
    query_embedding=query_embedding,
    top_k=6,
    document_ids=["paper_1", "paper_2", "paper_3", "paper_4"]
)
```

## API Usage Examples

```bash
# Search across all documents
curl -X POST "http://localhost:8000/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "top_k": 5
  }'

# Search in specific document (backward compatible)
curl -X POST "http://localhost:8000/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "top_k": 3,
    "document_id": "doc_123"
  }'

# Search across multiple documents (NEW)
curl -X POST "http://localhost:8000/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "top_k": 8,
    "document_ids": ["doc_1", "doc_2", "doc_3"]
  }'
```

## Function Signatures

### search_similar_chunks
```python
def search_similar_chunks(
    self, 
    query_embedding: Union[np.ndarray, List[float]], 
    top_k: int = 5,
    filter_criteria: Optional[Dict[str, Any]] = None,
    document_ids: Optional[Union[str, List[str]]] = None  # NEW PARAMETER
) -> List[Dict[str, Any]]
```

### search_by_query
```python
def search_by_query(
    self, 
    query: str, 
    embedder,  # EmbeddingGenerator instance
    top_k: int = 5,
    filter_criteria: Optional[Dict[str, Any]] = None,
    document_ids: Optional[Union[str, List[str]]] = None  # NEW PARAMETER
) -> List[Dict[str, Any]]
```

## ChromaDB Implementation Details

The multiple document ID filtering uses ChromaDB's `$in` operator:

```python
# Single document
filter_criteria = {"document_id": "doc_123"}

# Multiple documents  
filter_criteria = {"document_id": {"$in": ["doc_1", "doc_2", "doc_3"]}}
```

This leverages ChromaDB's built-in FAISS indexing for efficient filtering and similarity search.

## Benefits

1. **Flexible Filtering**: Search across any combination of documents
2. **Efficient**: Uses ChromaDB's optimized `$in` operator
3. **Backward Compatible**: Existing single document searches still work
4. **Scalable**: Performance scales well with number of documents
5. **Combined Filters**: Works alongside other metadata filters

## Use Cases

- **Multi-document Research**: Search across related research papers
- **User-specific Search**: Filter by user's uploaded documents
- **Category Filtering**: Search within document categories
- **Comparative Analysis**: Find similar content across specific documents
- **Incremental Search**: Progressively expand search scope