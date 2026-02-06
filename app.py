import streamlit as st
import requests, os, time, faiss, numpy as np
import speech_recognition as sr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

# =======================
# AUTH (simple demo)
# =======================

USERS = {"admin": "1234", "veera": "ai2026"}

def login():
    st.subheader("🔐 Login")
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

# =======================
# CONFIG
# =======================

API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3-70b-instruct"

# =======================
# EMBEDDING (cached)
# =======================

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# =======================
# VECTOR DB (FAST)
# =======================

DIM = 384
index = faiss.IndexFlatIP(DIM)
doc_chunks = []

def smart_chunks(text, size=400, overlap=50):
    words = text.split()
    out = []
    for i in range(0, len(words), size - overlap):
        out.append(" ".join(words[i:i + size]))
    return out

def add_to_memory(text):
    chunks = smart_chunks(text)
    vectors = embedder.encode(chunks, normalize_embeddings=True)
    index.add(vectors.astype("float32"))
    doc_chunks.extend(chunks)

def search_memory(q, k=4):
    if index.ntotal == 0:
        return ""
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
    _, ids = index.search(qv, k)
    return "\n".join(doc_chunks[i] for i in ids[0])

# =======================
# LLM CALL (safe)
# =======================

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
    return "⚠️ AI temporarily unavailable"

# =======================
# MULTI AGENT BRAIN
# =======================

AGENTS = {
    "planner": "Break the problem into clear steps",
    "researcher": "Provide insights and evidence",
    "architect": "Design scalable solution",
    "qa": "Find risks and edge cases"
}

def agent_workflow(q, context):
    outputs = {}
    for a, prompt in AGENTS.items():
        outputs[a] = call_ai([
            {"role": "system", "content": prompt},
            {"role": "user", "content": context + "\n" + q}
        ])

    merge = f"""
Planner: {outputs['planner']}
Researcher: {outputs['researcher']}
Architect: {outputs['architect']}
QA: {outputs['qa']}
"""
    return call_ai([
        {"role": "system", "content": "Combine into enterprise-grade answer"},
        {"role": "user", "content": merge}
    ])

# =======================
# CLOUD SAFE VOICE OUTPUT
# =======================

def speak_browser(text):
    components.html(f"""
    <script>
    var msg = new SpeechSynthesisUtterance({repr(text)});
    window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as s:
        audio = r.listen(s)
        return r.recognize_google(audio)

# =======================
# UI
# =======================

st.set_page_config("Enterprise AI Platform", layout="wide")
st.title("🚀 Veera Enterprise AI Platform")

tabs = st.tabs(["💬 AI Chat", "📄 Document Memory", "🎤 Voice AI", "⚙️ Automation"])

# ---------------- CHAT ----------------

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

# ---------------- DOC RAG ----------------

with tabs[1]:

    f = st.file_uploader("Upload PDF", type="pdf")

    if f:
        txt = "".join(p.extract_text() for p in PdfReader(f).pages if p.extract_text())
        add_to_memory(txt)
        st.success("📚 Document stored in vector memory!")

# ---------------- VOICE ----------------

with tabs[2]:

    st.info("🎙 Speak → AI answers with voice + text")

    if st.button("Speak now"):
        q = listen()
        st.write("You said:", q)

        mem = search_memory(q)
        ans = agent_workflow(q, mem)

        st.markdown(ans)
        speak_browser(ans)

# ---------------- AUTOMATION ----------------

with tabs[3]:

    st.subheader("🤖 Agent Automation Task")

    task = st.text_area("Describe task for agents")

    if st.button("Run Agents"):
        mem = search_memory(task)
        res = agent_workflow(task, mem)
        st.markdown(res)

# ---------------- LOGOUT ----------------

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
