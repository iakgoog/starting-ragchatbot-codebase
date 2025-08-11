# RAG System Query Processing Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend (script.js)
    participant A as FastAPI (app.py)
    participant R as RAG System
    participant AI as AI Generator
    participant TM as Tool Manager
    participant ST as Search Tool
    participant VS as Vector Store
    participant SM as Session Manager

    U->>F: Types query & hits Enter
    F->>F: Disable input, show loading
    F->>F: Add user message to chat
    
    F->>+A: POST /api/query {query, session_id}
    A->>A: Extract request data
    A->>A: Create session if needed
    
    A->>+R: rag_system.query(query, session_id)
    R->>R: Format prompt
    R->>SM: Get conversation history
    SM-->>R: Return history
    
    R->>+AI: generate_response(prompt, history, tools)
    AI->>AI: Build system prompt + context
    AI->>AI: Prepare API call with tools
    AI->>+AI: Call Claude API
    AI-->>AI: Claude decides to use search tool
    
    AI->>+TM: Execute tool requests
    TM->>+ST: search_course_content(query, filters)
    ST->>+VS: search_content(query, filters)
    VS->>VS: Semantic similarity search
    VS-->>-ST: Return search results
    ST->>ST: Format results & store sources
    ST-->>-TM: Return formatted content
    TM-->>-AI: Return tool results
    
    AI->>AI: Send tool results back to Claude
    AI->>AI: Get final synthesized response
    AI-->>-R: Return response text
    
    R->>TM: Get last sources
    TM-->>R: Return sources list
    R->>SM: Update conversation history
    R-->>-A: Return (response, sources)
    
    A->>A: Build QueryResponse
    A-->>-F: {answer, sources, session_id}
    
    F->>F: Update session_id if new
    F->>F: Remove loading spinner
    F->>F: Add assistant message + sources
    F->>F: Re-enable input, auto-scroll
    F-->>U: Display response
```

## Component Breakdown

### Frontend Layer
```
┌─────────────────────────────────────┐
│           Frontend (JS)             │
├─────────────────────────────────────┤
│ • Event handling (Enter/Click)      │
│ • Input validation & UI state       │
│ • Loading states & user feedback    │
│ • Message rendering (Markdown)      │
│ • Source display (collapsible)      │
│ • Session management (client-side)  │
└─────────────────────────────────────┘
```

### API Layer
```
┌─────────────────────────────────────┐
│          FastAPI Server             │
├─────────────────────────────────────┤
│ • CORS & middleware                 │
│ • Request validation (Pydantic)     │
│ • Session ID management             │
│ • Error handling & HTTP responses   │
│ • Static file serving               │
└─────────────────────────────────────┘
```

### RAG Orchestration
```
┌─────────────────────────────────────┐
│           RAG System                │
├─────────────────────────────────────┤
│ • Query preprocessing               │
│ • Component coordination            │
│ • Prompt engineering                │
│ • Response post-processing          │
│ • Source aggregation                │
└─────────────────────────────────────┘
```

### AI Processing
```
┌─────────────────────────────────────┐
│          AI Generator               │
├─────────────────────────────────────┤
│ • Claude API integration            │
│ • Tool execution orchestration     │
│ • Context management                │
│ • Response synthesis                │
│ • Temperature & parameter control   │
└─────────────────────────────────────┘
```

### Search & Retrieval
```
┌─────────────────────────────────────┐
│      Search Tools & Vector Store    │
├─────────────────────────────────────┤
│ • Semantic similarity search        │
│ • Course/lesson filtering           │
│ • Result ranking & formatting       │
│ • Source tracking                   │
│ • ChromaDB integration              │
└─────────────────────────────────────┘
```

### Data Flow Summary

```
User Input → Frontend → API → RAG System → AI Generator
                                    ↓
                            Tool Manager → Search Tool → Vector Store
                                    ↓
                            Claude API ← Tool Results ← Search Results
                                    ↓
                          Final Response → Sources → Frontend → User
```

## Key Features

- **Tool-Driven Search**: Claude autonomously decides when to search
- **Session Persistence**: Maintains conversation context across queries  
- **Source Attribution**: Tracks and displays relevant course materials
- **Error Resilience**: Graceful handling at each processing stage
- **Real-time Feedback**: Loading states and progressive UI updates