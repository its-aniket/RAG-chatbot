# RAG Pipeline with ChromaDB Storage

## Overview

This implementation provides a complete RAG (Retrieval Augmented Generation) pipeline with ChromaDB vector storage that uses FAISS indexing for fast similarity search.

## Components

### 1. Document Processing (`preprocessing.py`)
- Extracts text from PDF documents
- Splits text into semantically meaningful chunks
- Adds metadata for tracking and filtering

### 2. Embedding Generation (`embeddings.py`)
- Uses `intfloat/e5-small` model for generating embeddings
- Produces 384-dimensional vectors
- Optimized for both documents and queries

### 3. Vector Storage (`storage.py`)
- ChromaDB with FAISS indexing for fast retrieval
- Persistent storage with metadata filtering
- Cosine similarity search

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Key Dependencies

- `chromadb==0.4.15` - Vector database with FAISS
- `torch==2.0.1` - PyTorch for embeddings
- `transformers==4.34.0` - Hugging Face transformers
- `numpy==1.24.3` - Numerical computations

## Usage

### Complete Pipeline Test

```bash
cd backend
python test_complete_pipeline.py
```

### Individual Components

```python
from rag_pipeline.preprocessing import DocumentProcessor
from rag_pipeline.embeddings import EmbeddingGenerator
from rag_pipeline.storage import VectorStore

# Initialize components
doc_processor = DocumentProcessor()
embedder = EmbeddingGenerator()
vector_store = VectorStore(persist_directory="./chroma_db")

# Process document
chunks = doc_processor.process_document("path/to/document.pdf", "doc_id", "filename.pdf")

# Generate embeddings
embedded_chunks = embedder.embed_chunks(chunks)

# Store in ChromaDB
result = vector_store.store_embedded_chunks(embedded_chunks, document_id="doc_id")

# Search for similar content
similar_chunks = vector_store.search_by_query(
    query="your search query",
    embedder=embedder,
    top_k=5
)
```

## ChromaDB Features

### Automatic FAISS Indexing
ChromaDB automatically uses FAISS (Facebook AI Similarity Search) for efficient similarity search in high-dimensional vector spaces.

### Persistent Storage
All data is automatically persisted to disk and reloaded on startup.

### Metadata Filtering
Search can be filtered by document properties:

```python
# Search only specific document
results = vector_store.search_similar_chunks(
    query_embedding=embedding,
    filter_criteria={"document_id": "specific_doc"}
)
```

### Collection Management
- Get statistics: `vector_store.get_collection_stats()`
- Delete documents: `vector_store.delete_document("doc_id")`
- Clear collection: `vector_store.clear_collection()`

## Performance

### E5-Small Model
- **Embedding Dimension**: 384
- **Max Sequence Length**: 512 tokens
- **Performance**: Optimized for English text
- **Prefixes**: Uses "passage:" for documents, "query:" for search

### ChromaDB + FAISS
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Distance Metric**: Cosine similarity
- **Batch Processing**: Configurable batch sizes
- **Memory Efficient**: Streams large datasets

## File Structure

```
backend/
├── rag_pipeline/
│   ├── preprocessing.py     # Document processing
│   ├── embeddings.py        # E5-small embeddings
│   └── storage.py           # ChromaDB storage
├── test_complete_pipeline.py # Integration test
└── requirements.txt         # Dependencies
```

## API Integration

The storage module can be integrated with FastAPI endpoints for:
- Document upload and processing
- Real-time search and retrieval
- Collection management
- Analytics and statistics

## Next Steps

1. **Install ChromaDB**: `pip install chromadb==0.4.15`
2. **Run Tests**: Execute `python test_complete_pipeline.py`
3. **Integrate with API**: Add storage endpoints to FastAPI
4. **Add LLM**: Connect with OpenAI/other LLM for response generation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Issues**: E5-small will fallback to CPU automatically
3. **Storage Errors**: Check disk space and write permissions
4. **Memory Issues**: Reduce batch sizes in storage operations

### Performance Tuning

1. **Batch Size**: Adjust embedding batch size based on available memory
2. **Index Parameters**: ChromaDB automatically optimizes FAISS parameters
3. **Chunk Size**: Experiment with chunk sizes for your use case
4. **Top-K**: Limit search results to improve response times