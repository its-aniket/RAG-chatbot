# Persistent Storage Implementation

## Overview
This implementation adds persistent storage for both documents and chat memories to the RAG Chatbot system. Documents are only processed through the RAG pipeline once, and multiple chat sessions are supported with full conversation history.

## Database Schema

### Tables Created:

1. **processed_documents**
   - Tracks which documents have been processed
   - Stores file hash to detect identical content
   - Prevents reprocessing on app refresh

2. **chat_sessions**
   - Stores multiple chat sessions
   - Each session has a unique session_id and title
   - Supports soft deletion (is_active flag)

3. **chat_messages**
   - Stores individual messages within sessions
   - Supports both user and assistant messages
   - Sources are stored as JSON for assistant responses

4. **document_chunks** (for future use)
   - Tracks individual chunks created from documents
   - Links to ChromaDB embeddings

## Key Features

### Document Processing Optimization
- **Hash-based Detection**: Uses SHA256 hash of file content to detect duplicates
- **Database Tracking**: Marks documents as processed/failed in SQLite database
- **Skip Reprocessing**: Automatically skips documents that were already processed
- **Status Tracking**: Tracks processing status (processed, failed, processing)

### Multi-Session Chat Memory
- **Session Management**: Create, list, update, and delete chat sessions
- **Persistent History**: All messages are stored in database
- **Session Metadata**: Track creation time, last update, and activity status
- **Message Storage**: Supports rich messages with sources and metadata

## API Endpoints

### Chat Session Management
- `POST /chat/sessions` - Create new chat session
- `GET /chat/sessions` - List all active sessions
- `GET /chat/sessions/{session_id}` - Get chat history
- `POST /chat/sessions/{session_id}/messages` - Add message to session
- `PUT /chat/sessions/{session_id}/title` - Update session title
- `DELETE /chat/sessions/{session_id}` - Delete session (soft delete)
- `GET /chat/sessions/{session_id}/export` - Export session as JSON

### Document Processing
- Enhanced `/rag/process-document` endpoint now checks for existing documents
- Returns `already_processed` status if document was previously processed
- Tracks processed documents in database automatically

## Database Files
- **SQLite Database**: `rag_chatbot.db` (created automatically)
- **Location**: Backend root directory
- **Initialization**: Automatic on app startup

## Usage Examples

### Check if App Processes Duplicate Documents
1. Upload a PDF document
2. Restart the application
3. Upload the same PDF again
4. Should receive "already_processed" status instead of reprocessing

### Multiple Chat Sessions
1. Create a new chat session: `POST /chat/sessions`
2. Add messages to the session: `POST /chat/sessions/{session_id}/messages`
3. Create another session for a different conversation
4. Retrieve chat history: `GET /chat/sessions/{session_id}`
5. List all sessions: `GET /chat/sessions`

## Benefits

### Performance Improvements
- **Faster Startup**: No need to reprocess existing documents
- **Reduced Computation**: Skip embedding generation for known documents
- **Storage Efficiency**: Avoid duplicate document storage in ChromaDB

### User Experience
- **Session Continuity**: Resume conversations across app restarts
- **Chat Organization**: Multiple conversations organized by sessions
- **History Preservation**: Never lose conversation history
- **Session Management**: Easy to organize and manage multiple chats

## Technical Implementation

### Document Tracking Flow
1. User uploads document
2. System calculates file hash
3. Checks database for existing processed document with same hash
4. If found: Returns "already_processed" status
5. If new: Processes through RAG pipeline and marks as processed

### Chat Session Flow
1. Frontend creates new session via API
2. User and assistant messages stored in database
3. Session updated_at timestamp maintained automatically
4. Frontend can load previous sessions and continue conversations

## Database Services

### DocumentService
- `calculate_file_hash()` - Generate SHA256 hash
- `is_document_processed()` - Check if document exists
- `mark_document_processed()` - Mark successful processing
- `mark_document_failed()` - Mark failed processing
- `get_processed_documents()` - List all processed documents

### ChatService
- `create_chat_session()` - Create new session
- `get_chat_session()` - Retrieve session by ID
- `get_all_chat_sessions()` - List all active sessions
- `add_message()` - Add message to session
- `get_chat_history()` - Retrieve all messages in session
- `update_session_title()` - Change session title
- `delete_chat_session()` - Soft delete session

## Future Enhancements
- Session search and filtering
- Message editing and deletion
- Conversation export/import
- Session sharing and collaboration
- Advanced chat analytics
- Document version tracking