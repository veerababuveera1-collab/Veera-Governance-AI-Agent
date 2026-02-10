import streamlit as st
import os, time, faiss, numpy as np, logging, pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai
import hashlib

# ======================
# LOAD ENV
# ======================
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ======================
# MASTER PROMPT
# ======================
MASTER_PROMPT = """
You are part of the Veera Enterprise AI System,
a multi-agent digital workforce.

Core rules:
- Be accurate and professional.
- Use structured outputs.
- Avoid hallucinations.
- Focus on enterprise-grade responses.
"""

# ======================
# AUTH (secure hash)
# ======================
APP_USER = os.getenv("APP_USER", "veera")
APP_PASS_HASH = os.getenv(
    "APP_PASS_HASH",
    hashlib.sha256("ai2026".encode()).hexdigest()
)

def check_password(p):
    return hashlib.sha256(p.encode()).hexdigest() == APP_PASS_HASH

def login():
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == APP_USER and check_password(p):
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid credentials")

if "user" not in st.session_state:
    login()
    st.stop()

# ======================
# AI CONFIG
# ======================
def get_secret(name):
    return st.secrets.get(name) or os.getenv(name)

GROQ_API_KEY = get_secret("GROQ_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

if not GROQ_API_KEY and not GEMINI_API_KEY:
    st.error("No AI keys found.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None

# ======================
# EMBEDDINGS
# ======================
@st.cache_resource
def load_embedder():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model, model.get_sentence_embedding_dimension()

embedder, DIM = load_embedder()

# ======================
# MEMORY
# ======================
INDEX_FILE = "memory.index"
CHUNKS_FILE = "memory_chunks.pkl"
MAX_CHUNKS = 5000

def load_memory():
    try:
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            index = faiss.read_index(INDEX_FILE)
            with open(CHUNKS_FILE, "rb") as f:
                chunks = pickle.load(f)
            return index, chunks
    except:
        logging.warning("Memory load failed.")
    return faiss.IndexFlatIP(DIM), []

def save_memory(index, chunks):
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(chunks, f)
    except:
        logging.warning("Memory save failed.")

if "index" not in st.session_state:
    idx, ch = load_memory()
    st.session_state.index = idx
    st.session_state.doc_chunks = ch

def smart_chunks(text, size=400, overlap=60):
    words = text.split()
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size - overlap)
    ]

def add_to_memory(text):
    chunks = smart_chunks(text)
    vecs = embedder.encode(chunks, normalize_embeddings=True)
    st.session_state.index.add(vecs.astype("float32"))
    st.session_state.doc_chunks.extend(chunks)

    if len(st.session_state.doc_chunks) > MAX_CHUNKS:
        st.session_state.doc_chunks = st.session_state.doc_chunks[-MAX_CHUNKS:]

    save_memory(st.session_state.index, st.session_state.doc_chunks)

def search_memory(q, k=4):
    if st.session_state.index.ntotal == 0:
        return ""
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
    _, ids = st.session_state.index.search(qv, k)
    return "\n".join(
        st.session_state.doc_chunks[i]
        for i in ids[0]
        if i < len(st.session_state.doc_chunks)
    )

# ======================
# AI CALL
# ======================
def call_ai(system, user, retries=2):
    system_prompt = MASTER_PROMPT + "\n\n" + system
    prompt = f"{system_prompt}\n\n{user[:6000]}"

    # GROQ
    if groq_client:
        for _ in range(retries):
            try:
                resp = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user[:6000]}
                    ],
                )
                return resp.choices[0].message.content
            except Exception as e:
                logging.warning(f"Groq failed: {e}")
                time.sleep(1)

    # GEMINI
    if gemini_model:
        for _ in range(retries):
            try:
                resp = gemini_model.generate_content(prompt)
                return resp.text
            except Exception as e:
                logging.warning(f"Gemini failed: {e}")
                time.sleep(1)

    return "⚠️ AI service unavailable"

# ======================
# 12 AGENTS
# ======================
AGENTS = {
    "planner": "Break the task into modules.",
    "researcher": "Provide domain insights.",
    "architect": "Design system architecture.",
    "tech_stack": "Select appropriate technologies.",
    "backend": "Design backend APIs.",
    "frontend": "Design frontend UI.",
    "database": "Design database schema.",
    "devops": "Provide deployment plan.",
    "qa": "Create testing strategy.",
    "security": "Identify security risks.",
    "performance": "Suggest optimizations.",
    "documentation": "Generate final summary."
}

def agent_workflow(q, context):
    outputs = {}
    context = context[-4000:]
    progress = st.progress(0)
    step = 0
    total = len(AGENTS)

    for agent, role in AGENTS.items():
        result = call_ai(role, context + "\n" + q)
        outputs[agent] = result
        context += f"\n\n{agent.upper()}:\n{result}"

        step += 1
        progress.progress(step / total)

    merged = "\n\n".join(
        f"{name.upper()}:\n{outputs[name]}"
        for name in outputs
    )

    return call_ai(
        "Combine all agent outputs into one enterprise-grade response.",
        merged
    )

# ======================
# UI
# ======================
st.set_page_config("Veera Enterprise AI Platform", layout="wide")
st.title("🚀 Veera Enterprise AI Platform")

# Sidebar AI status
st.sidebar.markdown("### 🔑 AI Status")
st.sidebar.write("Groq:", "✅" if GROQ_API_KEY else "❌")
st.sidebar.write("Gemini:", "✅" if GEMINI_API_KEY else "❌")

# ======================
# CODE LAB
# ======================
st.sidebar.markdown("### 🧪 Code Lab")
test_prompt = st.sidebar.text_input("Test Prompt", "Hello")

if st.sidebar.button("Test Groq"):
    if groq_client:
        try:
            r = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": test_prompt}]
            )
            st.sidebar.success("Groq working")
            st.sidebar.write(r.choices[0].message.content)
        except:
            st.sidebar.error("Groq failed")

if st.sidebar.button("Test Gemini"):
    if gemini_model:
        try:
            r = gemini_model.generate_content(test_prompt)
            st.sidebar.success("Gemini working")
            st.sidebar.write(r.text)
        except:
            st.sidebar.error("Gemini failed")

tabs = st.tabs([
    "💬 AI Chat",
    "📄 Document Memory",
    "🎤 Voice Reply",
    "⚙️ Automation"
])

# CHAT
with tabs[0]:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask anything...")
    if q:
        mem = search_memory(q)
        ans = agent_workflow(q, mem)
        st.session_state.chat += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans}
        ]
        st.rerun()

# MEMORY
with tabs[1]:
    f = st.file_uploader("Upload PDF", type="pdf")
    if f:
        text = "".join(
            p.extract_text() for p in PdfReader(f).pages if p.extract_text()
        )
        add_to_memory(text)
        st.success("📚 Document stored")

# VOICE
with tabs[2]:
    voice_q = st.text_input("Ask for voice reply")
    if voice_q:
        mem = search_memory(voice_q)
        ans = agent_workflow(voice_q, mem)
        st.markdown(ans)
        components.html(f"""
        <script>
        const msg = new SpeechSynthesisUtterance({repr(ans)});
        window.speechSynthesis.speak(msg);
        </script>
        """, height=0)

# AUTOMATION
with tabs[3]:
    task = st.text_area("Describe task")
    if st.button("Run Automation"):
        mem = search_memory(task)
        result = agent_workflow(task, mem)
        st.markdown(result)

# LOGOUT
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
