# Multi-Agent Document Assistant

A high-performance, asynchronous Document Assistant that uses a **LangGraph-driven multi-agent system** to answer user queries with absolute correctness. By combining dense and sparse search indices (**Hybrid RAG**) with a self-correcting **verification loop**, this assistant prevents hallucinations and ensures every claim is backed by precise, valid citations from your uploaded files (PDF, DOCX, and TXT).

The frontend is a lightweight, dark-themed Glassmorphic dashboard built in vanilla HTML/CSS/JS that displays the conversation alongside real-time updates and active states of the multi-agent graph as it processes requests.

---

## 🚀 Key Features

*   **Multi-Agent Coordination (LangGraph)**: Uses an orchestrated network of specialized agents—**Planner, Retrieval, Summarizer, Verification, and Memory**—to process, verify, and store conversation turns.
*   **Self-Correcting Verification Loop**: The **Verification Agent** analyzes drafted answers against raw text sources. If it spots a hallucinated claim or a broken citation, it halts execution, provides detailed critique, and routes the state back to the **Planner Agent** to rebuild the response (up to 3 loopback attempts).
*   **Hybrid Vector & Keyword Ingestion**: Combines semantic embeddings (ChromaDB + OpenAI) with sparse keyword queries (BM25) using **Reciprocal Rank Fusion (RRF)** to retrieve the most contextually relevant chunks.
*   **Asynchronous Processing Pipeline**: Offloads document parsing, chunking, and embedding to a background **Arq worker** backed by **Redis**, ensuring API upload endpoints remain non-blocking.
*   **Intelligent Redis Caching**: Leverages Redis to cache exact query responses per conversation, serving instant subsequent responses while still logging accurate messaging sequences in PostgreSQL.
*   **Real-time Process Monitoring**: The dashboard displays real-time agent execution visualizers and debug logs, showcasing which node is active and what feedback is being generated.

---

## 🏗️ System Architecture

Below is the execution flow of a single chat request through the multi-agent system:

```mermaid
graph TD
    User([User Prompt]) --> CacheCheck{Redis Cache Hit?}
    
    %% Cache Hit Path
    CacheCheck -- Yes --> ReturnCache[Serve Response Instantly] --> End([End Chat Turn])
    
    %% Cache Miss Path
    CacheCheck -- No --> StateInit[Initialize Agent State]
    StateInit --> PlannerNode[Planner Agent]
    
    %% The Multi-Agent Loop
    PlannerNode --> RetrievalNode[Retrieval Agent<br>BM25 + Chroma + RRF]
    RetrievalNode --> SummarizerNode[Summarizer Agent<br>Draft Answer]
    SummarizerNode --> VerificationNode[Verification Agent<br>Check Hallucinations & Citations]
    
    %% Verification Branching
    VerificationNode --> Passed{Verification Passed?}
    Passed -- Yes --> MemoryNode[Memory Agent<br>Update Memory & Persist DB]
    Passed -- No --> RetryCheck{Retries < 3?}
    
    RetryCheck -- Yes --> Loopback[Loopback with Feedback] --> PlannerNode
    RetryCheck -- No --> MemoryNode
    
    MemoryNode --> UpdateCache[Cache Response in Redis]
    UpdateCache --> Response[Send Final Answer & Citations] --> End
    
    %% Styling
    classDef agent fill:#1a103c,stroke:#8a2be2,stroke-width:2px,color:#fff;
    classDef process fill:#0b0817,stroke:#333,stroke-width:1px,color:#a39eb9;
    classDef decision fill:#331533,stroke:#da70d6,stroke-width:1.5px,color:#fff;
    
    class PlannerNode,RetrievalNode,SummarizerNode,VerificationNode,MemoryNode agent;
    class StateInit,ReturnCache,UpdateCache process;
    class CacheCheck,Passed,RetryCheck decision;
```

---

## 🛠️ Tech Stack

*   **Backend Framework**: FastAPI (Asynchronous Python)
*   **Multi-Agent Orchestration**: LangGraph, LangChain
*   **Embedding & Search**: OpenAI Embeddings (`text-embedding-3-small`), ChromaDB (Dense Vector), Rank-BM25 (Sparse Index)
*   **Job Queue & Caching**: Redis, Arq (Asynchronous Python background workers)
*   **Database**: PostgreSQL (SQLAlchemy with `asyncpg` async driver)
*   **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism design, custom CSS custom variables), JavaScript (ES6, Fetch API)

---

## 📦 Getting Started

### Prerequisites

*   **Docker & Docker Compose** installed on your machine.
*   An **OpenAI API Key** (required for embeddings, synthesis, and verification).

---

### ⚡ Quick Start (Docker Compose)

The easiest way to run the entire stack (Database, Cache, Backend API, Worker, and Frontend) is using Docker Compose.

1.  **Clone the Repository** (or navigate to its root folder):
    ```bash
    git clone https://github.com/cherish787/Multi-Agent-Document.git
    cd Multi-Agent-Document
    ```

2.  **Create a `.env` file** in the root directory (or in the `backend` folder) and add your OpenAI API Key:
    ```env
    OPENAI_API_KEY="your-actual-openai-api-key-here"
    ```

3.  **Launch the Services**:
    ```bash
    docker-compose up --build
    ```

4.  **Access the Application**:
    *   **Frontend Interface**: open `frontend/index.html` directly in a browser (or serve it locally using a simple HTTP server).
    *   **Backend API Docs (Swagger UI)**: Navigate to `http://localhost:8000/docs`.

---

### 💻 Local Development Setup

If you prefer to run services manually without Docker, follow these steps:

#### 1. Start Services
Ensure you have local instances of **PostgreSQL** and **Redis** running on their default ports (`5432` and `6379`).

#### 2. Backend Installation & Run
1.  Navigate to the `backend` folder:
    ```bash
    cd backend
    ```
2.  Create and activate a python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Configure environmental variables in a `.env` file inside `backend/`:
    ```env
    PROJECT_NAME="Multi-Agent Document Assistant"
    DEBUG=True
    DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/document_assistant"
    REDIS_URL="redis://localhost:6379"
    OPENAI_API_KEY="your-actual-openai-api-key-here"
    ```
5.  Start the FastAPI application:
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```
6.  Open a new terminal session, activate the virtual environment in `backend/`, and start the **Arq Background Worker**:
    ```bash
    arq app.worker.WorkerSettings
    ```

#### 3. Frontend Run
Simply double-click `frontend/index.html` to open it in your browser, or spin up a simple static file server:
```bash
cd frontend
python3 -m http.server 3000
```
Then visit `http://localhost:3000`.

---

## 🧪 Testing

The codebase includes a suite of unit tests testing text chunking thresholds, RRF scores fusion, and graph routing mechanics. Run them directly from the `backend` directory:

```bash
cd backend
python -m unittest test_backend.py
```

---

## 🔌 API Reference Highlights

Here are the key API routes you can interact with:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/documents/upload` | Upload a text/PDF/Docx file. Dispatches an async chunking task. |
| `GET` | `/api/documents` | List metadata & processing states (`processing`, `completed`, `failed`) of all files. |
| `DELETE` | `/api/documents/{id}` | Delete a document from SQLite/Postgre, ChromaDB, and disk storage. |
| `POST` | `/api/chat` | Send a query to the agent network. Returns answers, citations, and agent execution trace. |
| `GET` | `/api/chat/conversations` | List all conversation sessions. |
| `GET` | `/api/chat/conversations/{id}/history` | Get chronologically ordered message logs for a specific conversation. |

---

## 👥 How the Multi-Agent System Works

1.  **Planner**: Examines the current conversation memory and the user's prompt to generate a optimized query plan. If the Verification agent has just rejected a draft, the Planner reads the failure feedback and adjusts its strategy.
2.  **Retrieval**: Queries the local vector store (ChromaDB + OpenAI embeddings) and the keyword index (BM25) using the Planner's strategy. It merges both result sets using Reciprocal Rank Fusion (RRF).
3.  **Summarizer**: Synthesizes the final answer using the retrieved context. It inserts proper citations linking facts back to their source files.
4.  **Verification**: Conducts fact-checking on the drafted response. It makes sure no claim is hallucinated and that every bracketed citation points to a real document chunk. If the check fails, it increments the retry count and loops back to the Planner with instructions.
5.  **Memory**: If the response passes verification, the Memory node updates the running chat state and saves context before sending the finalized response back.
