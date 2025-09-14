#!/usr/bin/env python3
"""
Process existing uploaded documents through RAG pipeline
"""

import sys
import os
from pathlib import Path
import uuid

# Add the parent directory to the path so we can import RAG components
sys.path.append(str(Path(__file__).parent))

from rag_pipeline.preprocessing import DocumentProcessor
from rag_pipeline.embeddings import EmbeddingGenerator
from rag_pipeline.storage import VectorStore

def process_existing_documents():
    """
    Process all existing uploaded documents through RAG pipeline
    """
    print("=== Processing Existing Documents ===")
    
    upload_dir = Path("uploads")
    if not upload_dir.exists():
        print("No uploads directory found")
        return
    
    # Initialize RAG components
    print("1. Initializing RAG components...")
    doc_processor = DocumentProcessor()
    embedder = EmbeddingGenerator()
    vector_store = VectorStore(persist_directory="./chroma_db")
    
    # Get current collection stats
    stats = vector_store.get_collection_stats()
    print(f"   Current ChromaDB stats: {stats}")
    
    # Find all PDF files in uploads
    pdf_files = list(upload_dir.glob("*.pdf"))
    print(f"\n2. Found {len(pdf_files)} PDF files to process:")
    
    processed_count = 0
    failed_count = 0
    
    for file_path in pdf_files:
        try:
            # Extract file ID and original filename from the saved filename
            # Format: uuid_originalname.pdf
            parts = file_path.name.split("_", 1)
            if len(parts) == 2:
                file_id = parts[0]
                original_filename = parts[1]
            else:
                # Generate new ID if format is unexpected
                file_id = str(uuid.uuid4())
                original_filename = file_path.name
            
            print(f"\n   Processing: {original_filename}")
            print(f"   File ID: {file_id}")
            print(f"   Path: {file_path}")
            
            # Check if document is already processed
            existing_chunks = vector_store.get_document_chunks(file_id)
            if existing_chunks:
                print(f"   ⚠️  Document already processed ({len(existing_chunks)} chunks). Skipping...")
                continue
            
            # Process document into chunks
            print("   📄 Processing document...")
            chunks = doc_processor.process_document(str(file_path), file_id, original_filename)
            print(f"   📄 Generated {len(chunks)} chunks")
            
            # Generate embeddings
            print("   🔢 Generating embeddings...")
            embedded_chunks = embedder.embed_chunks(chunks)
            embedded_count = len([c for c in embedded_chunks if c.get("embedding") is not None])
            print(f"   🔢 Embedded {embedded_count} chunks")
            
            # Store in vector database
            print("   💾 Storing in ChromaDB...")
            storage_result = vector_store.store_embedded_chunks(embedded_chunks, document_id=file_id)
            
            if storage_result.get("status") == "success":
                print(f"   ✅ Success: {storage_result.get('stored_chunks', 0)} chunks stored")
                processed_count += 1
            else:
                print(f"   ❌ Storage failed: {storage_result.get('message', 'Unknown error')}")
                failed_count += 1
                
        except Exception as e:
            print(f"   ❌ Error processing {file_path.name}: {str(e)}")
            failed_count += 1
    
    # Final stats
    print(f"\n3. Processing Complete:")
    print(f"   ✅ Successfully processed: {processed_count} documents")
    print(f"   ❌ Failed: {failed_count} documents")
    
    # Get updated collection stats
    final_stats = vector_store.get_collection_stats()
    print(f"   📊 Final ChromaDB stats: {final_stats}")
    
    print("\n=== Done ===")
    print("You should now be able to search across all your uploaded documents!")

if __name__ == "__main__":
    process_existing_documents()