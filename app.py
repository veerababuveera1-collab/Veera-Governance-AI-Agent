import streamlit as st
import requests, os, time, faiss, speech_recognition as sr, pyttsx3
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# =======================
# AUTH (simple demo)
# =======================

USERS = {"admin":"1234","veera":"ai2026"}

def login():
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if USERS.get(u)==p:
            st.session_state.user=u
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

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =======================
# VECTOR DB
# =======================

DIM = 384
index = faiss.IndexFlatL2(DIM)
doc_chunks = []

def add_to_memory(text):
    chunks = [text[i:i+500] for i in range(0,len(text),500)]
    emb = embedder.encode(chunks)
    index.add(np.array(emb).astype("float32"))
    doc_chunks.extend(chunks)

def search_memory(q):
    if index.ntotal==0: return ""
    qv = embedder.encode([q]).astype("float32")
    _, I = index.search(qv, 3)
    return "\n".join([doc_chunks[i] for i in I[0]])

# =======================
# LLM CALL
# =======================

def call_ai(messages):
    r = requests.post(URL, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"},
        json={"model":MODEL,"messages":messages},
        timeout=60
    )
    return r.json()["choices"][0]["message"]["content"]

# =======================
# MULTI AGENT WORKFLOW
# =======================

AGENTS = {
    "planner":"Break into steps",
    "researcher":"Give insights",
    "architect":"Design systems",
    "qa":"Find risks"
}

def agent_workflow(q, context):
    outs={}
    for a,p in AGENTS.items():
        outs[a]=call_ai([
            {"role":"system","content":p},
            {"role":"user","content":context+q}
        ])

    merge=f"""
Planner:{outs['planner']}
Researcher:{outs['researcher']}
Architect:{outs['architect']}
QA:{outs['qa']}
"""
    return call_ai([
        {"role":"system","content":"Combine into enterprise answer"},
        {"role":"user","content":merge}
    ])

# =======================
# VOICE
# =======================

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    r=sr.Recognizer()
    with sr.Microphone() as s:
        audio=r.listen(s)
        return r.recognize_google(audio)

# =======================
# UI
# =======================

st.set_page_config("Enterprise AI","wide")
st.title("🚀 Veera Enterprise AI Platform")

tabs=st.tabs(["💬 AI Chat","📄 Document Memory","🎤 Voice AI","⚙️ Automation"])

# ---------------- CHAT ----------------

with tabs[0]:
    if "chat" not in st.session_state: st.session_state.chat=[]
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q=st.chat_input("Ask anything...")
    if q:
        mem=search_memory(q)
        ans=agent_workflow(q, mem)
        st.session_state.chat += [{"role":"user","content":q},{"role":"assistant","content":ans}]
        st.rerun()

# ---------------- DOC RAG ----------------

with tabs[1]:
    f=st.file_uploader("Upload PDF",type="pdf")
    if f:
        txt="".join([p.extract_text() for p in PdfReader(f).pages])
        add_to_memory(txt)
        st.success("Stored in vector memory!")

# ---------------- VOICE ----------------

with tabs[2]:
    if st.button("🎙 Speak"):
        q=listen()
        st.write("You:",q)
        mem=search_memory(q)
        ans=agent_workflow(q, mem)
        st.write("AI:",ans)
        speak(ans)

# ---------------- AUTOMATION ----------------

with tabs[3]:
    st.subheader("🤖 Agent Workflow Tasks")
    task=st.text_area("Describe task")
    if st.button("Run Agents"):
        mem=search_memory(task)
        res=agent_workflow(task, mem)
        st.markdown(res)

# ---------------- LOGOUT ----------------

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
