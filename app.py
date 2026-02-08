import streamlit as st
import requests
import os
import time
import faiss
import numpy as np
import json
import hashlib
from datetime import datetime
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

APP_NAME = "Veera Enterprise AI"
MODEL = "meta-llama/llama-3-70b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    st.error("Missing OPENROUTER_API_KEY")
    st.stop()

DATA_DIR = "data"
MEMORY_INDEX = f"{DATA_DIR}/memory.index"
MEMORY_CHUNKS = f"{DATA_DIR}/chunks.json"
USERS_FILE = f"{DATA_DIR}/users.json"
LOG_FILE = f"{DATA_DIR}/activity.log"

os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# SECURITY
# =========================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        return json.load(open(USERS_FILE))
    return {"admin": hash_password("admin123")}

users = load_users()

def login():
    st.subheader("🔐 Secure Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in users and users[u] == hash_password(p):
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid credentials")

if "user" not in st.session_state:
    login()
    st.stop()

# =========================
# LOGGING
# =========================

def log_activity(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} | {msg}\n")

# =========================
# AI CORE
# =========================

def call_ai(messages, max_tokens=1000):
    try:
        r = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens
            },
            timeout=60
        )
        data = r.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
    except:
        pass
    return "AI service unavailable"

# =========================
# AGENT CLASS
# =========================

class Agent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt

    def run(self, task, context=""):
        return call_ai([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context + "\n" + task}
        ])

# =========================
# AGENT DEFINITIONS
# =========================

planner = Agent("Planner",
    "Break the goal into clear step-by-step tasks.")

researcher = Agent("Researcher",
    "Retrieve relevant context and information.")

executor = Agent("Executor",
    "Execute the task and produce results.")

coder = Agent("Coder",
    "Write clean, runnable code if needed.")

critic = Agent("Critic",
    "Review the output. Say OK if correct, or FIX with reason.")

supervisor = Agent("Supervisor",
    "Decide whether to CONTINUE, FIX, or FINISH.")

# =========================
# MEMORY SYSTEM
# =========================

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
DIM = 384

def load_memory():
    if os.path.exists(MEMORY_INDEX):
        index = faiss.read_index(MEMORY_INDEX)
        chunks = json.load(open(MEMORY_CHUNKS))
    else:
        index = faiss.IndexFlatIP(DIM)
        chunks = []
    return index, chunks

index, doc_chunks = load_memory()

def search_memory(q, k=4):
    if index.ntotal == 0:
        return ""
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
    _, ids = index.search(qv, k)
    return "\n".join(doc_chunks[i] for i in ids[0] if i < len(doc_chunks))

# =========================
# AUTONOMOUS EXECUTION LOOP
# =========================

def autonomous_run(goal, max_steps=4):
    log = []
    context = ""

    # Step 1: Planning
    plan = planner.run(goal)
    log.append(f"🧠 PLAN:\n{plan}")

    steps = plan.split("\n")

    for i, step in enumerate(steps[:max_steps]):
        log.append(f"⚙️ STEP {i+1}: {step}")

        memory = search_memory(step)
        research = researcher.run(step, memory)

        result = executor.run(step, research)
        review = critic.run(result)

        log.append(f"📊 RESULT:\n{result}")
        log.append(f"🔍 REVIEW:\n{review}")

        decision = supervisor.run(review)

        log.append(f"🧭 DECISION: {decision}")

        if "FINISH" in decision.upper():
            break

    return "\n\n".join(log)

# =========================
# UI
# =========================

st.set_page_config(APP_NAME, layout="wide")
st.title("🚀 Veera Enterprise AI Platform")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "AI Chat", "Autonomous Mode", "Activity"]
)

# =========================
# DASHBOARD
# =========================

if menu == "Dashboard":
    st.metric("Memory Chunks", len(doc_chunks))
    st.metric("User", st.session_state.user)

# =========================
# CHAT
# =========================

elif menu == "AI Chat":
    q = st.text_input("Ask AI")
    if q:
        mem = search_memory(q)
        ans = executor.run(q, mem)
        st.write(ans)
        log_activity("Chat used")

# =========================
# AUTONOMOUS MODE
# =========================

elif menu == "Autonomous Mode":
    st.subheader("🤖 Autonomous AI Task Execution")

    goal = st.text_area("Describe your goal")

    if st.button("Run Autonomous AI") and goal:
        output = autonomous_run(goal)
        st.text_area("Execution Log", output, height=500)
        log_activity("Autonomous run executed")

# =========================
# ACTIVITY
# =========================

elif menu == "Activity":
    if os.path.exists(LOG_FILE):
        st.text(open(LOG_FILE).read())

# =========================
# LOGOUT
# =========================

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
