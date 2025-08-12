
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Quick start**: `chmod +x run.sh && ./run.sh`
- **Manual start**: `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Dependencies**: `uv sync` (requires uv package manager)

### Environment Setup
- Create `.env` file in root with `ANTHROPIC_API_KEY=your_api_key_here`
- Application runs on http://localhost:8000
- API docs available at http://localhost:8000/docs

## Architecture Overview

### RAG System Design
This is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about course materials using:
- **FastAPI backend** (`backend/app.py`) with CORS and static file serving
- **RAGSystem orchestrator** (`backend/rag_system.py`) that coordinates all components
- **ChromaDB vector storage** for semantic search of course content
- **Anthropic Claude Sonnet 4** for AI response generation with tool-calling capabilities
- **Tool-based search architecture** where the AI uses search tools rather than direct vector retrieval

### Core Components Flow
1. **Document Processing** (`document_processor.py`): Parses course documents into structured Course/Lesson objects with chunked content
2. **Vector Storage** (`vector_store.py`): ChromaDB client managing course metadata and content chunks with SentenceTransformer embeddings
3. **AI Generator** (`ai_generator.py`): Claude API integration with tool-calling support and conversation history
4. **Search Tools** (`search_tools.py`): CourseSearchTool that the AI can call to search the vector database
5. **Session Management** (`session_manager.py`): Tracks conversation history per session

### Key Architecture Decisions
- **Tool-based search**: AI uses CourseSearchTool rather than direct vector access, enabling more sophisticated search strategies
- **Component separation**: Each major function (processing, storage, generation, search) is isolated in separate modules
- **Configuration centralization**: All settings managed through `config.py` with environment variable support
- **Startup document loading**: Documents from `../docs` folder automatically processed on server start

### Frontend Integration
- Simple HTML/CSS/JS frontend served as static files from `/frontend`
- API endpoints: `/api/query` (main chat), `/api/courses` (analytics)
- Session-based conversation tracking with sources display

### Document Structure
- Expects course documents in `docs/` folder (PDF, DOCX, TXT)
- Processes into Course → Lessons → Content chunks hierarchy
- Automatically detects existing courses to avoid reprocessing
- always use uv to run the server do not use pip directly
- always use uv to manage all dependencies
- use uv to run Python files