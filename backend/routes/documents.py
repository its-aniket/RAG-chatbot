from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
from pathlib import Path
import PyPDF2
from io import BytesIO

router = APIRouter(prefix="/documents", tags=["documents"])

# Configure upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document for RAG processing
    """
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are allowed"
            )
        
        # Check file size (limit to 10MB)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=400,
                detail="File size must be less than 10MB"
            )
        
        # Validate PDF content
        try:
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            
            if num_pages == 0:
                raise HTTPException(
                    status_code=400,
                    detail="PDF file appears to be empty or corrupted"
                )
            
            # Extract some text to verify it's readable
            first_page = pdf_reader.pages[0]
            text_sample = first_page.extract_text()
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid PDF file: {str(e)}"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process through RAG pipeline
        try:
            # Import RAG components
            from rag_pipeline.preprocessing import DocumentProcessor
            from rag_pipeline.embeddings import EmbeddingGenerator
            from rag_pipeline.storage import VectorStore
            
            # Initialize RAG components
            doc_processor = DocumentProcessor()
            embedder = EmbeddingGenerator()
            vector_store = VectorStore(persist_directory="./chroma_db")
            
            # Process document into chunks
            chunks = doc_processor.process_document(str(file_path), file_id, file.filename)
            
            # Generate embeddings
            embedded_chunks = embedder.embed_chunks(chunks)
            embedded_count = len([c for c in embedded_chunks if c.get("embedding") is not None])
            
            # Store in vector database
            storage_result = vector_store.store_embedded_chunks(embedded_chunks, document_id=file_id)
            
            rag_status = "success"
            rag_message = f"Processed {embedded_count} chunks into vector database"
            
        except Exception as rag_error:
            rag_status = "partial"
            rag_message = f"File uploaded but RAG processing failed: {str(rag_error)}"
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF uploaded successfully",
                "file_id": file_id,
                "filename": file.filename,
                "pages": num_pages,
                "size_bytes": len(content),
                "text_preview": text_sample[:200] + "..." if len(text_sample) > 200 else text_sample,
                "rag_status": rag_status,
                "rag_message": rag_message
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}"
        )

@router.get("/list")
async def list_uploaded_documents():
    """
    List all uploaded documents
    """
    try:
        documents = []
        for file_path in UPLOAD_DIR.glob("*.pdf"):
            file_stats = file_path.stat()
            # Extract file_id from filename (format: uuid_originalname.pdf)
            parts = file_path.name.split("_", 1)
            file_id = parts[0] if len(parts) > 1 else "unknown"
            original_name = parts[1] if len(parts) > 1 else file_path.name
            
            documents.append({
                "file_id": file_id,
                "filename": original_name,
                "uploaded_at": file_stats.st_ctime,
                "size_bytes": file_stats.st_size
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "documents": documents,
                "total": len(documents)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while listing documents: {str(e)}"
        )

@router.delete("/delete/{file_id}")
async def delete_document(file_id: str):
    """
    Delete an uploaded document by file_id
    """
    try:
        # Find file with this ID
        matching_files = list(UPLOAD_DIR.glob(f"{file_id}_*.pdf"))
        
        if not matching_files:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        # Delete the file
        file_path = matching_files[0]
        file_path.unlink()
        
        # TODO: Remove from vector database
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Document deleted successfully",
                "file_id": file_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while deleting the document: {str(e)}"
        )