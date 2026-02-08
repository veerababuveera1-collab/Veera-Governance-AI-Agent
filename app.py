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
# MODELS
# ======================

MODELS = {
    "fast": "stepfun/step-3.5-flash",        # free
    "main": "moonshotai/kimi-k2.5",          # powerful
    "coding": "qwen/qwen3-coder-next",       # best coding
    "reasoning": "anthropic/claude-opus-4.6",# deep reasoning
    "fallback": "arcee/trinity-large-preview" # free fallback
}

MASTER_PROMPT = """
You are Veera-Governance-AI-Agent.

Follow these rules:
1. Write production-ready, clean, secure code.
2. Fix errors automatically.
3. Optimize performance.
4. Ensure the code runs without crashes.
"""

# ======================
# SESSION STATE
# ======================

if "chat" not in st.session_state:
    st.session_state.chat = []

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
# INTENT ROUTER
# ======================

def detect_intent(text):
    t = text.lower()
    if any(x in t for x in ["error", "bug", "code", "api"]):
        return "technical"
    if any(x in t for x in ["price", "plan"]):
        return "sales"
    return "general"

def route_model(intent):
    if intent == "technical":
        return MODELS["coding"]
    return MODELS["main"]

# ======================
# SAFE AI CALL WITH FALLBACK
# ======================

def call_ai(messages, model, max_tokens=450, retries=2):

    def send(model_name):
        r = requests.post(
            URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://veeratech.ai",
                "X-Title": "Veera Governance AI"
            },
            json={
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens
            },
            timeout=60
        )
        return r.json()

    for _ in range(retries):
        try:
            data = send(model)
            if "choices" in data:
                return data["choices"][0]["message"]["content"]

            if "error" in data:
                err = data["error"].get("message", "").lower()
                if any(x in err for x in ["credit", "token", "insufficient"]):
                    break
                return f"⚠️ {data['error'].get('message')}"

        except:
            time.sleep(1)

    # fallback
    try:
        data = send(MODELS["fallback"])
        if "choices" in data:
            return "⚠️ Free model used:\n\n" + data["choices"][0]["message"]["content"]
    except:
        pass

    return "❌ AI unavailable"

# ======================
# CHAT AGENT
# ======================

def agent_workflow(q, context):
    intent = detect_intent(q)
    model = route_model(intent)

    messages = [
        {"role": "system", "content": MASTER_PROMPT},
        {"role": "user", "content": context + "\n" + q}
    ]
    return call_ai(messages, model)

# ======================
# CODE AUTO-FIX PIPELINE
# ======================

def code_pipeline(project_desc):

    # 1. Architecture
    arch = call_ai([
        {"role": "system", "content": "You are a senior architect"},
        {"role": "user", "content": project_desc + "\nCreate system architecture"}
    ], MODELS["reasoning"])

    # 2. Code generation
    code = call_ai([
        {"role": "system", "content": "Write production-ready code"},
        {"role": "user", "content": project_desc}
    ], MODELS["coding"])

    # 3. Fix bugs
    fixed = call_ai([
        {"role": "system", "content": "Fix all bugs and missing imports"},
        {"role": "user", "content": code}
    ], MODELS["coding"])

    # 4. Optimize
    optimized = call_ai([
        {"role": "system", "content": "Optimize performance and stability"},
        {"role": "user", "content": fixed}
    ], MODELS["coding"])

    # 5. Review
    reviewed = call_ai([
        {"role": "system", "content": "Review and return final production-ready code"},
        {"role": "user", "content": optimized}
    ], MODELS["reasoning"])

    return arch, reviewed

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
    "💬 AI Chat",
    "📄 Document Memory",
    "🎤 Voice",
    "⚙️ Automation",
    "💻 Code Lab"
])

# ---------- CHAT ----------

with tabs[0]:
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask something...")
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
    f = st.file_uploader("Upload PDF", type="pdf")
    if f:
        text = "".join(p.extract_text() for p in PdfReader(f).pages if p.extract_text())
        add_to_memory(text)
        st.success("Document stored")

# ---------- VOICE ----------

with tabs[2]:
    vq = st.text_input("Voice question")
    if vq:
        mem = search_memory(vq)
        ans = agent_workflow(vq, mem)
        st.markdown(ans)
        speak(ans)

# ---------- AUTOMATION ----------

with tabs[3]:
    task = st.text_area("Describe task for AI agents")
    if st.button("Run"):
        mem = search_memory(task)
        res = agent_workflow(task, mem)
        st.markdown(res)

# ---------- CODE LAB AUTO PIPELINE ----------

with tabs[4]:
    st.subheader("💻 One-Click Auto-Fix Code Lab")

    project = st.text_area(
        "Describe system",
        placeholder="Build production-ready bus booking system"
    )

    if st.button("🚀 Generate Error-Free Code"):
        arch, final_code = code_pipeline(project)

        st.markdown("### 🧱 Architecture")
        st.code(arch)

        st.markdown("### ✅ Final Production Code")
        st.code(final_code, language="python")

# ======================
# LOGOUT
# ======================

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
