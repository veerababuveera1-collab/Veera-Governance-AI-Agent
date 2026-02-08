import streamlit as st
import requests, os, time, faiss, numpy as np
import sqlite3, bcrypt, zipfile, io
from datetime import datetime
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# ======================
# DATABASE SETUP
# ======================

conn = sqlite3.connect("veera_ai.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password BLOB
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    role TEXT,
    message TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    title TEXT,
    content TEXT,
    created_at TEXT
)
""")

conn.commit()

# ======================
# AUTH
# ======================

def create_user(username, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except:
        return False

def authenticate(username, password):
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    row = c.fetchone()
    if row and bcrypt.checkpw(password.encode(), row[0]):
        return True
    return False

def login_ui():
    st.subheader("🔐 Login / Signup")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if authenticate(u, p):
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username", key="signup_user")
        p = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Create Account"):
            if create_user(u, p):
                st.success("Account created. Please login.")
            else:
                st.error("Username already exists.")

if "user" not in st.session_state:
    login_ui()
    st.stop()

# ======================
# CONFIG
# ======================

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    st.error("Missing OPENROUTER_API_KEY")
    st.stop()

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3-70b-instruct"

# ======================
# EMBEDDINGS
# ======================

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================
# VECTOR MEMORY
# ======================

DIM = 384
index = faiss.IndexFlatIP(DIM)
doc_chunks = []

def smart_chunks(text, size=400, overlap=60):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))
    return chunks

def load_documents_from_db():
    c.execute("SELECT content FROM documents")
    rows = c.fetchall()
    for row in rows:
        add_to_memory(row[0], store=False)

def add_to_memory(text, store=True):
    chunks = smart_chunks(text)
    vecs = embedder.encode(chunks, normalize_embeddings=True)
    index.add(vecs.astype("float32"))
    doc_chunks.extend(chunks)

    if store:
        c.execute("INSERT INTO documents (content) VALUES (?)", (text,))
        conn.commit()

def search_memory(q, k=4):
    if index.ntotal == 0:
        return ""
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
    _, ids = index.search(qv, k)
    return "\n".join(doc_chunks[i] for i in ids[0] if i < len(doc_chunks))

if "docs_loaded" not in st.session_state:
    load_documents_from_db()
    st.session_state.docs_loaded = True

# ======================
# LLM CALL
# ======================

def call_ai(messages, retries=2):
    for _ in range(retries):
        try:
            r = requests.post(
                URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={"model": MODEL, "messages": messages},
                timeout=60
            )
            return r.json()["choices"][0]["message"]["content"]
        except:
            time.sleep(1)
    return "⚠️ AI service unavailable"

# ======================
# AGENTS
# ======================

AGENTS = {
    "planner": "Break into clear steps",
    "researcher": "Give insights and facts",
    "architect": "Design scalable solution",
    "qa": "Find risks and flaws"
}

def agent_workflow(q, context):
    outputs = {}
    for name, role in AGENTS.items():
        outputs[name] = call_ai([
            {"role": "system", "content": role},
            {"role": "user", "content": context + "\n" + q}
        ])
    merged = "\n".join(f"{k.upper()}: {v}" for k, v in outputs.items())
    return call_ai([
        {"role": "system", "content": "Combine into enterprise-grade answer"},
        {"role": "user", "content": merged}
    ])

CODE_AGENTS = {
    "planner": "Break the project into modules and steps",
    "architect": "Design system architecture and tech stack",
    "backend_dev": "Write backend APIs and logic",
    "frontend_dev": "Create UI and frontend structure",
    "db_engineer": "Design database schema and queries",
    "qa": "Find bugs, edge cases, and logic flaws",
    "security": "Identify security vulnerabilities",
    "optimizer": "Improve performance and scalability"
}

def run_code_lab(task, context=""):
    outputs = {}
    for name, role in CODE_AGENTS.items():
        outputs[name] = call_ai([
            {"role": "system", "content": role},
            {"role": "user", "content": context + "\n" + task}
        ])
    final = call_ai([
        {"role": "system", "content": "Combine into a full production-ready software project."},
        {"role": "user", "content": str(outputs)}
    ])
    return final

# ======================
# ZIP GENERATOR
# ======================

def create_project_zip(title, content):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("project.txt", content)
        zf.writestr("README.md", f"# {title}\n\nGenerated by Veera AI Code Lab.")
    zip_buffer.seek(0)
    return zip_buffer

# ======================
# UI
# ======================

st.set_page_config("Veera Enterprise AI Platform", layout="wide")
st.title("🚀 Veera Enterprise AI Platform")

tabs = st.tabs([
    "💬 AI Chat",
    "📄 Document Memory",
    "🎤 Voice Reply",
    "⚙️ Automation",
    "💻 Code Lab",
    "📁 My Projects",
    "📊 Analytics"
])

# -------- ANALYTICS --------
with tabs[6]:
    st.subheader("📊 Platform Analytics")

    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM chats")
    total_chats = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM documents")
    total_docs = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM projects")
    total_projects = c.fetchone()[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users", total_users)
    col2.metric("Chats", total_chats)
    col3.metric("Documents", total_docs)
    col4.metric("Projects", total_projects)

    # Chart
    labels = ["Users", "Chats", "Documents", "Projects"]
    values = [total_users, total_chats, total_docs, total_projects]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Platform Usage Overview")
    st.pyplot(fig)

# -------- LOGOUT --------
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
