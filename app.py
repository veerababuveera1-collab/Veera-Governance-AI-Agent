import streamlit as st
import requests, os, time, faiss, numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

# ======================
# PAGE CONFIG
# ======================

st.set_page_config("Veera-Governance-AI-Agent", layout="wide")
st.title("🏛️ Veera-Governance-AI-Agent")

# ======================
# AUTH
# ======================

USERS = {"admin": "1234", "veera": "ai2026"}

def login():
    st.subheader("🔐 Secure Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if USERS.get(u) == p:
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid credentials")

if "user" not in st.session_state:
    login()
    st.stop()

# ======================
# CONFIG
# ======================

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    st.error("Missing OPENROUTER_API_KEY")
    st.stop()

URL = "https://openrouter.ai/api/v1/chat/completions"

# ======================
# MULTI-MODEL CONFIG
# ======================

MODELS = {
    "fast": "stepfun/step-3.5-flash",
    "main": "moonshotai/kimi-k2.5",
    "coding": "qwen/qwen3-coder-next",
    "reasoning": "anthropic/claude-opus-4.6",
    "fallback": "arcee/trinity-large-preview"
}

MASTER_PROMPT = """
You are Veera-Governance-AI-Agent,
an intelligent AI system for governance, administration, and citizen services.

Core responsibilities:
1. Assist citizens and officials.
2. Provide accurate, policy-based responses.
3. Use document memory when available.
4. Be professional, clear, and secure.
5. Do not hallucinate unknown facts.
6. If unsure, escalate to human support.
"""

# ======================
# SESSION STATE
# ======================

if "chat" not in st.session_state:
    st.session_state.chat = []

if "current_step" not in st.session_state:
    st.session_state.current_step = 0

if "generated_files" not in st.session_state:
    st.session_state.generated_files = {}

if "project_architecture" not in st.session_state:
    st.session_state.project_architecture = None

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

def chunk_text(text, size=400, overlap=60):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-overlap)]

def add_to_memory(text):
    chunks = chunk_text(text)
    vecs = embedder.encode(chunks, normalize_embeddings=True)
    index.add(vecs.astype("float32"))
    doc_chunks.extend(chunks)

def search_memory(q, k=4):
    if index.ntotal == 0:
        return ""
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
    _, ids = index.search(qv, k)
    return "\n".join(doc_chunks[i] for i in ids[0])

# ======================
# INTENT + MODEL ROUTER
# ======================

def detect_intent(text):
    t = text.lower()

    if any(x in t for x in ["error", "bug", "api", "code", "crash"]):
        return "technical"
    if any(x in t for x in ["plan", "price", "cost", "difference"]):
        return "sales"
    if any(x in t for x in ["refund", "charged", "billing"]):
        return "billing"
    if any(x in t for x in ["delete", "remove account"]):
        return "deletion"

    return "general"

def route_model(intent):
    if intent == "technical":
        return MODELS["coding"]
    elif intent == "billing":
        return MODELS["fast"]
    elif intent == "sales":
        return MODELS["main"]
    elif intent == "general":
        return MODELS["main"]
    else:
        return MODELS["fallback"]

# ======================
# SAFE LLM CALL
# ======================

def call_ai(messages, model, max_tokens=800, retries=2):
    for _ in range(retries):
        try:
            r = requests.post(
                URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://veeratech.ai",
                    "X-Title": "Veera Governance AI"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens
                },
                timeout=60
            )

            data = r.json()

            if "choices" in data:
                return data["choices"][0]["message"]["content"]

            if "error" in data:
                return f"⚠️ AI Error: {data['error'].get('message','Unknown')}"

            return "⚠️ Empty response"

        except Exception:
            time.sleep(1)

    return "❌ AI service unavailable"

# ======================
# AGENT WORKFLOW
# ======================

def agent_workflow(q, context):
    intent = detect_intent(q)
    model = route_model(intent)

    messages = [
        {"role": "system", "content": MASTER_PROMPT},
        {"role": "user", "content": context + "\n" + q}
    ]

    return call_ai(messages, model=model, max_tokens=700)

# ======================
# VOICE
# ======================

def speak(text):
    components.html(f"""
    <script>
    const msg = new SpeechSynthesisUtterance({repr(text)});
    speechSynthesis.speak(msg);
    </script>
    """, height=0)

# ======================
# CODE LAB HELPERS
# ======================

FILE_ORDER = [
    "database.py",
    "auth.py",
    "booking_engine.py",
    "seat_management.py",
    "payment.py",
    "admin.py",
    "app.py"
]

def next_file():
    idx = st.session_state.current_step - 1
    return FILE_ORDER[idx] if idx < len(FILE_ORDER) else None

# ======================
# UI TABS
# ======================

tabs = st.tabs([
    "💬 Citizen & Officer Chat",
    "📄 Policy Document Memory",
    "🎤 Voice Assistant",
    "⚙️ Automation Agent",
    "💻 Code Lab"
])

# ---------- CHAT ----------

with tabs[0]:
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask governance question...")
    if q:
        mem = search_memory(q)
        ans = agent_workflow(q, mem)
        st.session_state.chat += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans}
        ]
        st.rerun()

# ---------- DOCUMENT MEMORY ----------

with tabs[1]:
    f = st.file_uploader("Upload policy PDF", type="pdf")
    if f:
        text = "".join(p.extract_text() for p in PdfReader(f).pages if p.extract_text())
        add_to_memory(text)
        st.success("📚 Document stored in governance memory")

# ---------- VOICE ----------

with tabs[2]:
    vq = st.text_input("Ask by voice text")
    if vq:
        mem = search_memory(vq)
        ans = agent_workflow(vq, mem)
        st.markdown(ans)
        speak(ans)

# ---------- AUTOMATION ----------

with tabs[3]:
    task = st.text_area("Describe administrative task")
    if st.button("Run Automation"):
        mem = search_memory(task)
        res = agent_workflow(task, mem)
        st.markdown(res)

# ---------- CODE LAB ----------

with tabs[4]:
    st.subheader("💻 Government IT Code Lab")

    project = st.text_area(
        "Describe system to build",
        placeholder="Build digital land record system"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("🧱 Architecture"):
            arch = call_ai([
                {"role": "system", "content": MASTER_PROMPT},
                {"role": "user", "content": project + "\nCreate folder structure"}
            ], model=MODELS["reasoning"], max_tokens=600)
            st.code(arch)

    with c2:
        if st.button("📄 Generate Module"):
            code = call_ai([
                {"role": "system", "content": MASTER_PROMPT},
                {"role": "user", "content": project + "\nGenerate production code"}
            ], model=MODELS["coding"], max_tokens=1000)
            st.code(code, language="python")

    with c3:
        if st.button("🔁 Review System"):
            review = call_ai([
                {"role": "system", "content": MASTER_PROMPT},
                {"role": "user", "content": project + "\nReview system"}
            ], model=MODELS["reasoning"], max_tokens=600)
            st.markdown(review)

# ======================
# LOGOUT
# ======================

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
