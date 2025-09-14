#!/usr/bin/env python3
"""
Quick test script to verify the ChromaDB search fix
"""

import sys
import os
from pathlib import Path

# Add the rag_pipeline directory to Python path
sys.path.append(str(Path(__file__).parent / "rag_pipeline"))

def test_search_fix():
    """Test the fixed search functionality"""
    try:
        print("=== ChromaDB Search Fix Test ===")
        
        from storage import VectorStore
        from embeddings import EmbeddingGenerator
        
        # Initialize components
        print("1. Initializing components...")
        embedder = EmbeddingGenerator()
        vector_store = VectorStore(
            persist_directory="./test_chroma_db",
            collection_name="test_search_fix"
        )
        
        # Check if we have any documents in the collection
        stats = vector_store.get_collection_stats()
        print(f"2. Collection has {stats['total_chunks']} chunks")
        
        if stats['total_chunks'] == 0:
            print("   No documents in collection. Please run the full pipeline test first.")
            return False
        
        # Test 1: Search all documents (this was failing before)
        print("\n3. Testing search all documents (previously failed)...")
        try:
            all_results = vector_store.search_by_query(
                query="machine learning",
                embedder=embedder,
                top_k=3
                # No document_ids = search all
            )
            print(f"   ✓ SUCCESS: Found {len(all_results)} results across all documents")
            
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
            return False
        
        # Test 2: Search with specific document (this was working)
        print("\n4. Testing search specific document...")
        try:
            if stats['unique_documents'] > 0:
                # Get a document ID from the collection
                sample_results = vector_store.collection.get(limit=1, include=["metadatas"])
                if sample_results["metadatas"]:
                    doc_id = sample_results["metadatas"][0].get("document_id")
                    if doc_id:
                        specific_results = vector_store.search_by_query(
                            query="machine learning",
                            embedder=embedder,
                            top_k=2,
                            document_ids=doc_id
                        )
                        print(f"   ✓ SUCCESS: Found {len(specific_results)} results in document {doc_id[:8]}...")
                    else:
                        print("   ⚠ SKIP: No document_id in metadata")
                else:
                    print("   ⚠ SKIP: No metadata found")
            else:
                print("   ⚠ SKIP: No documents in collection")
                
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
            return False
        
        # Test 3: Search with multiple documents
        print("\n5. Testing search multiple documents...")
        try:
            # Get multiple document IDs
            sample_results = vector_store.collection.get(limit=10, include=["metadatas"])
            doc_ids = list(set([
                meta.get("document_id") 
                for meta in sample_results["metadatas"] 
                if meta.get("document_id")
            ]))
            
            if len(doc_ids) > 0:
                multi_results = vector_store.search_by_query(
                    query="machine learning",
                    embedder=embedder,
                    top_k=3,
                    document_ids=doc_ids[:3]  # Use up to 3 document IDs
                )
                print(f"   ✓ SUCCESS: Found {len(multi_results)} results across {len(doc_ids[:3])} documents")
            else:
                print("   ⚠ SKIP: No document IDs found")
                
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
            return False
        
        # Test 4: Search with empty filter (edge case)
        print("\n6. Testing edge cases...")
        try:
            empty_filter_results = vector_store.search_similar_chunks(
                query_embedding=embedder.embed_query("test"),
                top_k=2,
                filter_criteria=None,  # Explicitly None
                document_ids=None      # Explicitly None
            )
            print(f"   ✓ SUCCESS: Edge case with None filters found {len(empty_filter_results)} results")
            
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
            return False
        
        print("\n=== All Tests Passed! ===")
        print("The ChromaDB search fix is working correctly.")
        print("\nFix Summary:")
        print("- ✓ Search all documents (no filtering) now works")
        print("- ✓ Search specific documents still works") 
        print("- ✓ Search multiple documents still works")
        print("- ✓ Edge cases with None values handled properly")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Please ensure ChromaDB and other dependencies are installed.")
        return False
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_search_fix()
    exit(0 if success else 1)