#!/usr/bin/env python3
"""
Test the complete RAG pipeline with LLM integration
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_llm_rag_pipeline():
    """
    Test the complete RAG pipeline with LLM response generation
    """
    print("=== RAG + LLM Integration Test ===")
    
    try:
        # Check if Groq API key is set
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("❌ GROQ_API_KEY environment variable not set")
            print("Please set your Groq API key:")
            print("export GROQ_API_KEY=your_key_here")
            return False
        
        print("✅ Groq API key found")
        
        # Import components
        from rag_pipeline.storage import VectorStore
        from rag_pipeline.embeddings import EmbeddingGenerator
        from rag_pipeline.llm_generator import LLMGenerator
        
        print("✅ All modules imported successfully")
        
        # Initialize components
        print("\n1. Initializing components...")
        vector_store = VectorStore()
        embedder = EmbeddingGenerator()
        llm_generator = LLMGenerator()
        
        print("✅ All components initialized")
        
        # Check if we have documents
        stats = vector_store.get_collection_stats()
        print(f"📊 Collection stats: {stats}")
        
        if stats.get('total_chunks', 0) == 0:
            print("❌ No documents in collection. Please upload some documents first.")
            return False
        
        # Test query
        test_query = "what is ecology"
        print(f"\n2. Testing query: '{test_query}'")
        
        # Search for relevant chunks
        print("🔍 Searching for relevant chunks...")
        results = vector_store.search_by_query(
            query=test_query,
            embedder=embedder,
            top_k=3
        )
        
        print(f"✅ Found {len(results)} relevant chunks")
        
        if results:
            for i, result in enumerate(results[:2]):
                filename = result['metadata'].get('filename', 'Unknown')
                score = result['similarity_score']
                print(f"   Result {i+1}: {filename} (Score: {score:.3f})")
        
        # Generate LLM response
        print("\n3. Generating LLM response...")
        llm_response = llm_generator.generate_response(
            query=test_query,
            
            retrieved_chunks=results
        )
        
        print("✅ LLM response generated successfully")
        
        # Display results
        print(f"\n4. Results:")
        print(f"Query: {test_query}")
        print(f"Model used: {llm_response['model_used']}")
        print(f"Token usage: {llm_response['token_usage']}")
        print(f"Sources: {len(llm_response['sources'])} documents")
        print(f"\nResponse:\n{llm_response['response']}")
        
        print("\n✅ Complete RAG+LLM pipeline test successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        print("Please install missing dependencies: pip install groq")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_groq_connection():
    """
    Test Groq API connection only
    """
    print("=== Testing Groq API Connection ===")
    
    try:
        from rag_pipeline.llm_generator import LLMGenerator
        
        # Test with minimal setup
        llm = LLMGenerator()
        print("✅ Groq LLM initialized successfully")
        
        # Test with sample data
        sample_chunks = [
            {
                "chunk_id": "test_1",
                "text": "The sky is blue due to Rayleigh scattering of light.",
                "similarity_score": 0.95,
                "metadata": {"filename": "physics.pdf", "chunk_index": 1}
            }
        ]
        
        response = llm.generate_response(
            query="Why is the sky blue?",
            retrieved_chunks=sample_chunks
        )
        
        print("✅ Test response generated:")
        print(f"Model: {response['model_used']}")
        print(f"Tokens: {response['token_usage']['total_tokens']}")
        print(f"Response: {response['response'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Groq connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test Groq connection first
    if test_groq_connection():
        print("\n" + "="*50 + "\n")
        # Test full pipeline
        test_llm_rag_pipeline()
    else:
        print("\nPlease fix Groq API setup before testing full pipeline")