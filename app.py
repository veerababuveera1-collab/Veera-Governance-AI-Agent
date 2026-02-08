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
# MODEL CONFIG
# ======================

MODELS = {
    "Auto (Smart Routing)": "auto",
    "Fast Free Model": "stepfun/step-3.5-flash",
    "Coding Model": "qwen/qwen3-coder-next",
    "Reasoning Model": "arcee/trinity-large-preview",
    "Premium Model": "moonshotai/kimi-k2.5"
}

DEFAULT_FALLBACK = "stepfun/step-3.5-flash"

st.sidebar.header("🧠 AI Model Settings")
model_choice = st.sidebar.selectbox(
    "Select AI Model",
    list(MODELS.keys())
)
selected_model = MODELS[model_choice]

# ======================
# MASTER PROMPT
# ======================

MASTER_PROMPT = """
You are Veera-Governance-AI-Agent.
You assist citizens, officers, and administrators.

Rules:
- Provide accurate, clear, practical answers.
- Use document memory when available.
- Do not hallucinate unknown facts.
- If unsure, suggest human escalation.
"""

# ======================
# SESSION STATE
# ======================

if "chat" not in st.session_state:
    st.session_state.chat = []

if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

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
# SAFE AI CALL WITH FALLBACK
# ======================

def call_ai(messages, model, max_tokens=350):
    models_to_try = []

    if model == "auto":
        models_to_try = [
            MODELS["Fast Free Model"],
            MODELS["Coding Model"],
            MODELS["Reasoning Model"]
        ]
    else:
        models_to_try = [model]

    models_to_try.append(DEFAULT_FALLBACK)

    for m in models_to_try:
        try:
            r = requests.post(
                URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": m,
                    "messages": messages,
                    "max_tokens": max_tokens
                },
                timeout=60
            )

            data = r.json()

            if "choices" in data:
                return data["choices"][0]["message"]["content"]

        except:
            continue

    return "❌ AI unavailable"

# ======================
# AGENT WORKFLOW
# ======================

def agent_workflow(q, context):
    messages = [
        {"role": "system", "content": MASTER_PROMPT},
        {"role": "user", "content": context + "\n" + q}
    ]
    return call_ai(messages, model=selected_model, max_tokens=350)

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
# UI TABS
# ======================

tabs = st.tabs([
    "💬 Chat",
    "📄 Document Memory",
    "🎤 Voice Assistant",
    "⚙️ Automation",
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
        st.success("📚 Document stored in memory")

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

# ---------- CODE LAB (AUTO-FIX PIPELINE) ----------
with tabs[4]:
    st.subheader("💻 Auto-Fix Code Lab")

    project = st.text_area(
        "Describe system to build",
        placeholder="Build citizen complaint management system"
    )

    if st.button("🚀 Generate Error-Free Code"):
        # Step 1: Generate code
        code = call_ai([
            {"role": "system", "content": "You are a senior software engineer"},
            {"role": "user", "content": project + "\nGenerate production-ready Python code"}
        ], model=MODELS["Coding Model"], max_tokens=800)

        # Step 2: Fix bugs
        fixed = call_ai([
            {"role": "system", "content": "Fix all bugs and missing imports"},
            {"role": "user", "content": code}
        ], model=MODELS["Coding Model"], max_tokens=600)

        # Step 3: Optimize
        optimized = call_ai([
            {"role": "system", "content": "Optimize for performance and stability"},
            {"role": "user", "content": fixed}
        ], model=MODELS["Coding Model"], max_tokens=600)

        # Step 4: Final review
        final = call_ai([
            {"role": "system", "content": "Review and output final clean production code only"},
            {"role": "user", "content": optimized}
        ], model=MODELS["Coding Model"], max_tokens=700)

        st.session_state.generated_code = final

    if st.session_state.generated_code:
        st.code(st.session_state.generated_code, language="python")

# ======================
# LOGOUT
# ======================

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
