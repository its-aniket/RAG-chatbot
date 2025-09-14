# RAG Chatbot

## Overview
The RAG (Retrieval-Augmented Generation) Chatbot is an advanced AI-powered chatbot application designed to provide accurate and interactive responses by leveraging multi-document search and large language models (LLMs). The project integrates a robust backend with a modern frontend to deliver a seamless user experience.

## Features
- **Multi-document Search**: Extracts accurate responses from multiple documents.
- **LLM Integration**: Provides responses with proper markdown formatting.
- **Advanced Citation Tools**: Includes collapsible sources, hover tooltips, and quality badges.
- **Persistent Storage**: Prevents duplicate document processing and supports multiple chat sessions.
- **Modern UI**: Built with React and Tailwind CSS, featuring hidden scrollbars for a clean look.
- **ChromaDB Integration**: Optimized vector storage for efficient document retrieval.

## Technologies Used
### Backend
- **Framework**: FastAPI
- **Database**: SQLite with SQLAlchemy ORM
- **Vector Storage**: ChromaDB
- **Language Model**: Groq LLM

### Frontend
- **Framework**: React
- **Styling**: Tailwind CSS

## Installation
### Prerequisites
- Python 3.9+
- Node.js 16+

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the backend server:
   ```bash
   uvicorn app:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the frontend server:
   ```bash
   npm start
   ```

## Usage
1. Start both the backend and frontend servers.
2. Open the frontend in your browser at `http://localhost:3000`.
3. Upload documents and interact with the chatbot.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.