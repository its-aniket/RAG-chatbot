"""
Database services for document and chat management
"""
from sqlalchemy.orm import Session
from .models import ProcessedDocument, ChatSession, ChatMessage, DocumentChunk
from typing import List, Optional
import hashlib
import json
from datetime import datetime

class DocumentService:
    """Service for managing processed documents"""
    
    @staticmethod
    def calculate_file_hash(file_content: bytes) -> str:
        """Calculate SHA256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    @staticmethod
    def is_document_processed(db: Session, filename: str, file_hash: str) -> bool:
        """Check if document has already been processed"""
        document = db.query(ProcessedDocument).filter(
            ProcessedDocument.filename == filename,
            ProcessedDocument.file_hash == file_hash,
            ProcessedDocument.status == "processed"
        ).first()
        return document is not None
    
    @staticmethod
    def mark_document_processed(
        db: Session, 
        filename: str, 
        file_hash: str, 
        file_size: int, 
        chunk_count: int = 0
    ) -> ProcessedDocument:
        """Mark document as processed"""
        document = ProcessedDocument(
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            chunk_count=chunk_count,
            status="processed"
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        return document
    
    @staticmethod
    def get_processed_documents(db: Session) -> List[ProcessedDocument]:
        """Get all processed documents"""
        return db.query(ProcessedDocument).filter(
            ProcessedDocument.status == "processed"
        ).all()
    
    @staticmethod
    def mark_document_failed(db: Session, filename: str, file_hash: str, file_size: int):
        """Mark document processing as failed"""
        document = ProcessedDocument(
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            status="failed"
        )
        db.add(document)
        db.commit()

class ChatService:
    """Service for managing chat sessions and messages"""
    
    @staticmethod
    def create_chat_session(db: Session, title: str = "New Chat") -> ChatSession:
        """Create a new chat session"""
        session = ChatSession(title=title)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    
    @staticmethod
    def get_chat_session(db: Session, session_id: str) -> Optional[ChatSession]:
        """Get chat session by ID"""
        return db.query(ChatSession).filter(
            ChatSession.session_id == session_id,
            ChatSession.is_active == True
        ).first()
    
    @staticmethod
    def get_all_chat_sessions(db: Session) -> List[ChatSession]:
        """Get all active chat sessions"""
        return db.query(ChatSession).filter(
            ChatSession.is_active == True
        ).order_by(ChatSession.updated_at.desc()).all()
    
    @staticmethod
    def update_session_title(db: Session, session_id: str, title: str) -> Optional[ChatSession]:
        """Update chat session title"""
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id
        ).first()
        if session:
            session.title = title
            session.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(session)
        return session
    
    @staticmethod
    def delete_chat_session(db: Session, session_id: str) -> bool:
        """Delete chat session (soft delete)"""
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id
        ).first()
        if session:
            session.is_active = False
            db.commit()
            return True
        return False
    
    @staticmethod
    def add_message(
        db: Session, 
        session_id: str, 
        message_type: str, 
        content: str, 
        sources: List[dict] = None
    ) -> ChatMessage:
        """Add message to chat session"""
        sources_json = json.dumps(sources) if sources else None
        
        message = ChatMessage(
            session_id=session_id,
            message_type=message_type,
            content=content,
            sources_json=sources_json
        )
        db.add(message)
        
        # Update session updated_at
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id
        ).first()
        if session:
            session.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(message)
        return message
    
    @staticmethod
    def get_chat_history(db: Session, session_id: str) -> List[ChatMessage]:
        """Get chat history for a session"""
        return db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp.asc()).all()
    
    @staticmethod
    def format_message_for_api(message: ChatMessage) -> dict:
        """Format database message for API response"""
        sources = None
        if message.sources_json:
            try:
                sources = json.loads(message.sources_json)
            except json.JSONDecodeError:
                sources = None
        
        return {
            "id": message.id,
            "type": message.message_type,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "sources": sources
        }