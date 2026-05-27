/* ==========================================================================
   MULTI-AGENT DOCUMENT ASSISTANT - FRONTEND LOGIC
   ========================================================================== */

const API_BASE = "http://localhost:8000";

// Global App State
let activeConversationId = null;
let isQuerying = false;
let pollingInterval = null;

// DOM Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const documentList = document.getElementById("document-list");
const btnRefreshDocs = document.getElementById("btn-refresh-docs");
const conversationList = document.getElementById("conversation-list");
const activeChatTitle = document.getElementById("active-chat-title");
const chatStatus = document.getElementById("chat-status");
const btnNewChat = document.getElementById("btn-new-chat");
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const queryInput = document.getElementById("query-input");
const btnSend = document.getElementById("btn-send");
const monitorBadge = document.getElementById("monitor-badge");
const workflowLogs = document.getElementById("workflow-logs");

// Flowchart Nodes
const nodes = {
    "Planner Agent": document.getElementById("node-planner"),
    "Retrieval Agent": document.getElementById("node-retrieval"),
    "Summarizer Agent": document.getElementById("node-summarizer"),
    "Verification Agent": document.getElementById("node-verification"),
    "Memory Agent": document.getElementById("node-memory")
};
const arrowFeedback = document.getElementById("arrow-feedback");

// --- STARTUP / INITIALIZATION ---
document.addEventListener("DOMContentLoaded", () => {
    loadDocuments();
    loadConversations();
    setupUploadHandlers();
    
    // Refresh buttons
    btnRefreshDocs.addEventListener("click", loadDocuments);
    btnNewChat.addEventListener("click", startNewChat);
    
    // Chat Submit
    chatForm.addEventListener("submit", handleChatSubmit);
});

// --- DOCUMENT INGESTION HANDLERS ---

function setupUploadHandlers() {
    // Click dropzone to select file
    dropzone.addEventListener("click", () => fileInput.click());
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadFiles(e.target.files);
        }
    });
    
    // Drag & Drop
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        }, false);
    });
    
    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        }, false);
    });
    
    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            uploadFiles(files);
        }
    });
}

async function uploadFiles(files) {
    for (let file of files) {
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            updateChatStatus("Uploading file...", true);
            const response = await fetch(`${API_BASE}/api/documents/upload`, {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) {
                const err = await response.json();
                alert(`Upload failed for ${file.name}: ${err.detail}`);
            } else {
                loadDocuments();
            }
        } catch (error) {
            console.error("Upload error:", error);
            alert(`Network error uploading file: ${file.name}`);
        }
    }
    updateChatStatus("System Idle", false);
}

async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/api/documents`);
        if (!response.ok) return;
        
        const docs = await response.json();
        renderDocuments(docs);
        
        // Trigger status polling if any document is processing
        const hasProcessing = docs.some(d => d.status === "processing");
        if (hasProcessing && !pollingInterval) {
            pollingInterval = setInterval(loadDocuments, 3000);
        } else if (!hasProcessing && pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
    } catch (error) {
        console.error("Failed to load documents:", error);
    }
}

function renderDocuments(docs) {
    if (docs.length === 0) {
        documentList.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-box-open"></i>
                <p>No documents uploaded yet</p>
            </div>
        `;
        return;
    }
    
    documentList.innerHTML = docs.map(doc => `
        <div class="doc-item" data-id="${doc.id}">
            <div class="doc-info">
                <i class="fa-solid ${getFileIcon(doc.filename)}"></i>
                <div class="doc-name-wrapper" style="overflow:hidden;">
                    <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
                    <span class="status-pill status-${doc.status}">${doc.status}</span>
                </div>
            </div>
            <button class="btn-delete-doc" onclick="deleteDocument('${doc.id}', event)" title="Delete document">
                <i class="fa-solid fa-trash-can"></i>
            </button>
        </div>
    `).join("");
}

function getFileIcon(filename) {
    const ext = filename.split(".").pop().toLowerCase();
    if (ext === "pdf") return "fa-file-pdf";
    if (ext === "docx") return "fa-file-word";
    return "fa-file-lines";
}

async function deleteDocument(docId, event) {
    event.stopPropagation();
    if (!confirm("Are you sure you want to delete this document from the library and database?")) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/documents/${docId}`, {
            method: "DELETE"
        });
        if (response.ok) {
            loadDocuments();
        }
    } catch (error) {
        console.error("Failed to delete document:", error);
    }
}

// --- CONVERSATION HISTORY LOGIC ---

async function loadConversations() {
    try {
        const response = await fetch(`${API_BASE}/api/chat/conversations`);
        if (!response.ok) return;
        const convs = await response.json();
        renderConversations(convs);
    } catch (error) {
        console.error("Failed to load conversations:", error);
    }
}

function renderConversations(convs) {
    if (convs.length === 0) {
        conversationList.innerHTML = `
            <div class="empty-state">
                <p>No chat history</p>
            </div>
        `;
        return;
    }
    
    conversationList.innerHTML = convs.map(c => `
        <div class="conv-item ${c.id === activeConversationId ? 'active-conv' : ''}" onclick="selectConversation('${c.id}')">
            <div class="conv-info">
                <i class="fa-solid fa-message"></i>
                <div class="conv-name" title="${c.title}">${c.title}</div>
            </div>
        </div>
    `).join("");
}

async function selectConversation(convId) {
    if (isQuerying) return;
    activeConversationId = convId;
    
    // Highlight selected item
    document.querySelectorAll(".conv-item").forEach(item => {
        item.classList.remove("active-conv");
    });
    const activeItem = document.querySelector(`.conv-item[onclick="selectConversation('${convId}')"]`);
    if (activeItem) activeItem.classList.add("active-conv");
    
    // Fetch History
    try {
        updateChatStatus("Loading history...", true);
        const response = await fetch(`${API_BASE}/api/chat/conversations/${convId}/history`);
        if (!response.ok) return;
        
        const history = await response.json();
        chatMessages.innerHTML = "";
        
        if (history.length === 0) {
            renderWelcomeMessage();
        } else {
            history.forEach(msg => {
                appendMessageBubble(msg.role, msg.content, msg.citations);
            });
            scrollToBottom();
        }
        
        // Find title
        const convResponse = await fetch(`${API_BASE}/api/chat/conversations`);
        const convs = await convResponse.json();
        const active = convs.find(c => c.id === convId);
        if (active) {
            activeChatTitle.textContent = active.title;
        }
        
    } catch (error) {
        console.error("Failed to load history:", error);
    } finally {
        updateChatStatus("System Idle", false);
    }
}

function startNewChat() {
    if (isQuerying) return;
    activeConversationId = null;
    activeChatTitle.textContent = "New Discussion";
    chatMessages.innerHTML = "";
    renderWelcomeMessage();
    clearAgentVisualizer();
    loadConversations();
}

function renderWelcomeMessage() {
    chatMessages.innerHTML = `
        <div class="message assistant-msg welcome-msg">
            <div class="msg-avatar">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div class="msg-bubble">
                <h2>Hello! I'm your Intelligent Multi-Agent Assistant.</h2>
                <p>I am backed by a pipeline of 5 specialized agents that work collaboratively to find precise answers grounded in your documents:</p>
                <ul class="welcome-agent-list">
                    <li><strong><i class="fa-solid fa-compass"></i> Planner Agent</strong>: Strategizes the optimal retrieval queries.</li>
                    <li><strong><i class="fa-solid fa-magnifying-glass"></i> Retrieval Agent</strong>: Conducts hybrid vector & keyword searches.</li>
                    <li><strong><i class="fa-solid fa-feather-pointed"></i> Summarizer Agent</strong>: Formulates citation-grounded answers.</li>
                    <li><strong><i class="fa-solid fa-shield-halved"></i> Verification Agent</strong>: Screens responses to prevent hallucinations.</li>
                    <li><strong><i class="fa-solid fa-database"></i> Memory Agent</strong>: Syncs long-term key insights and context.</li>
                </ul>
                <p>To start, upload some documents in the sidebar and ask me anything!</p>
            </div>
        </div>
    `;
}

// --- CHAT INTERACTION LOGIC ---

async function handleChatSubmit(e) {
    e.preventDefault();
    if (isQuerying) return;
    
    const query = queryInput.value.trim();
    if (!query) return;
    
    // Clear Input
    queryInput.value = "";
    
    // Lock submit UI
    isQuerying = true;
    btnSend.disabled = true;
    btnSend.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
    
    // Add user message to UI
    appendMessageBubble("user", query);
    scrollToBottom();
    
    // Clear visualizer logs
    clearAgentVisualizer();
    setMonitorState("Active");
    updateChatStatus("Multi-Agent LangGraph executing...", true);
    
    // Assemble payload
    const payload = {
        query: query,
        conversation_id: activeConversationId
    };
    
    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server error");
        }
        
        const data = await response.json();
        
        // Update active conversation
        if (!activeConversationId) {
            activeConversationId = data.conversation_id;
            loadConversations();
            activeChatTitle.textContent = `Discussion - ${new Date().toLocaleTimeString()}`;
        }
        
        // Sequentially animate workflow steps for high-fidelity interactive tracer
        await animateAgentWorkflow(data.agent_steps, data.answer, data.citations);
        
    } catch (error) {
        console.error("Chat Error:", error);
        appendMessageBubble("assistant", `An error occurred: ${error.message}`);
        updateChatStatus("System Error", false);
        setMonitorState("Inactive");
        scrollToBottom();
    } finally {
        isQuerying = false;
        btnSend.disabled = false;
        btnSend.innerHTML = '<i class="fa-solid fa-paper-plane"></i>';
    }
}

// --- AGENT TRACER ANIMATIONS ---

function clearAgentVisualizer() {
    Object.values(nodes).forEach(node => {
        node.className = "visual-node";
    });
    arrowFeedback.style.display = "none";
    workflowLogs.innerHTML = `
        <div class="log-empty-state">
            <i class="fa-solid fa-terminal"></i>
            <p>Waiting for agent workflow logs...</p>
        </div>
    `;
}

function setMonitorState(state) {
    monitorBadge.textContent = state;
    if (state === "Active") {
        monitorBadge.className = "active-badge active";
    } else {
        monitorBadge.className = "active-badge";
    }
}

function updateChatStatus(text, loading) {
    chatStatus.innerHTML = `<i class="fa-solid ${loading ? 'fa-spinner fa-spin' : 'fa-circle-check'}"></i> ${text}`;
    if (loading) {
        chatStatus.className = "chat-status status-loading";
    } else {
        chatStatus.className = "chat-status";
    }
}

// Animate Agent progression visually in Sidebar
async function animateAgentWorkflow(steps, answer, citations) {
    if (!steps || steps.length === 0) {
        // Fallback if no steps returned
        appendMessageBubble("assistant", answer, citations);
        setMonitorState("Inactive");
        updateChatStatus("System Idle", false);
        scrollToBottom();
        return;
    }
    
    workflowLogs.innerHTML = ""; // Clear empty state
    
    // Standard delay (500ms) to give the user a high-fidelity visual tracing feeling!
    const delay = (ms) => new Promise(res => setTimeout(res, ms));
    
    for (let idx = 0; idx < steps.length; idx++) {
        const step = steps[idx];
        const agentName = step.agent;
        
        // 1. Highlight Node in Graph Flowchart
        Object.keys(nodes).forEach(name => {
            if (nodes[name]) {
                if (name === agentName) {
                    nodes[name].className = "visual-node active-node";
                } else if (nodes[name].classList.contains("active-node")) {
                    nodes[name].className = "visual-node completed-node";
                }
            }
        });
        
        // Special case: Loopback feedback display
        if (step.action.includes("Verification failed")) {
            arrowFeedback.style.display = "flex";
        } else if (step.action.includes("Verification passed")) {
            arrowFeedback.style.display = "none";
        }
        
        // 2. Append Log Timeline card
        const logCard = document.createElement("div");
        logCard.className = "log-card";
        
        // Accent edge depending on validation
        if (step.action.includes("failed")) {
            logCard.style.borderLeftColor = "var(--status-error)";
        } else if (step.action.includes("passed") || agentName === "Memory Agent") {
            logCard.style.borderLeftColor = "var(--status-success)";
        } else if (agentName === "Planner Agent") {
            logCard.style.borderLeftColor = "var(--accent-purple)";
        } else if (agentName === "Retrieval Agent") {
            logCard.style.borderLeftColor = "var(--accent-blue)";
        } else {
            logCard.style.borderLeftColor = "var(--accent-pink)";
        }
        
        const timestamp = new Date(step.timestamp).toLocaleTimeString();
        logCard.innerHTML = `
            <div class="log-card-header">
                <span class="log-agent">${agentName}</span>
                <span class="log-time">${timestamp}</span>
            </div>
            <div class="log-card-body">${step.action}</div>
        `;
        
        workflowLogs.appendChild(logCard);
        workflowLogs.scrollTop = workflowLogs.scrollHeight;
        
        await delay(600); // 600ms visual wait
    }
    
    // Complete all nodes
    Object.keys(nodes).forEach(name => {
        if (nodes[name]) nodes[name].className = "visual-node completed-node";
    });
    
    // Render final Q&A bubble
    appendMessageBubble("assistant", answer, citations);
    scrollToBottom();
    
    setMonitorState("Inactive");
    updateChatStatus("System Idle", false);
}

// --- MESSAGE RENDERING ---

function appendMessageBubble(role, content, citations) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role === "user" ? "user-msg" : "assistant-msg"}`;
    
    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.innerHTML = role === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    
    // Parse formatting (e.g. bold lists and newlines)
    const formattedContent = formatMessageText(content);
    bubble.innerHTML = `<p>${formattedContent}</p>`;
    
    // Inject expandable citations if present and role is assistant
    if (role === "assistant" && citations && citations.length > 0) {
        const uniqueId = `citation-panel-${Math.random().toString(36).substr(2, 9)}`;
        
        const citationsWrapper = document.createElement("div");
        citationsWrapper.className = "citations-wrapper";
        citationsWrapper.innerHTML = `
            <button class="citations-toggle" onclick="toggleCitations('${uniqueId}', this)">
                <i class="fa-solid fa-chevron-down"></i> Citations (${citations.length})
            </button>
            <div class="citations-panel" id="${uniqueId}">
                ${citations.map((c, i) => `
                    <div class="citation-card">
                        <div class="citation-card-header">
                            <i class="fa-solid fa-file-signature"></i>
                            <span>[Citation ${i + 1}] Source: ${c.filename}</span>
                        </div>
                        <div class="citation-text">"${c.text}"</div>
                    </div>
                `).join("")}
            </div>
        `;
        bubble.appendChild(citationsWrapper);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(bubble);
    
    chatMessages.appendChild(messageDiv);
}

function formatMessageText(text) {
    if (!text) return "";
    // Basic Markdown formatting helper
    return text
        .replace(/\n/g, "<br>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>");
}

// Global Citations Toggle helper
window.toggleCitations = function(id, button) {
    const panel = document.getElementById(id);
    if (panel) {
        const isActive = panel.classList.toggle("active");
        button.classList.toggle("active", isActive);
    }
};

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
