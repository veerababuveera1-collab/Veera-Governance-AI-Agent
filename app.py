import streamlit as st
import os, time, logging, zipfile, tempfile, hashlib, faiss, pickle
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

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

Rules:
- Be accurate and professional.
- Use structured outputs.
- No hallucinations.
- Enterprise-grade responses only.
"""

# ======================
# AUTH
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
# AI CALL
# ======================
def call_ai(system, user, retries=2):
    system_prompt = MASTER_PROMPT + "\n\n" + system
    prompt = f"{system_prompt}\n\n{user[:6000]}"

    # GROQ
    if groq_client:
        for _ in range(retries):
            try:
                r = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user[:6000]}
                    ]
                )
                return r.choices[0].message.content
            except Exception as e:
                logging.error(f"Groq error: {e}")
                time.sleep(1)

    # GEMINI fallback
    if gemini_model:
        for _ in range(retries):
            try:
                r = gemini_model.generate_content(prompt)
                return r.text
            except Exception as e:
                logging.error(f"Gemini error: {e}")
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

def agent_workflow(task):
    context = task
    outputs = {}

    for agent, role in AGENTS.items():
        result = call_ai(role, context)
        outputs[agent] = result
        context += "\n" + result

    merged = "\n\n".join(
        f"{name.upper()}:\n{outputs[name]}"
        for name in outputs
    )

    return call_ai("Combine into structured enterprise response.", merged)

# ======================
# MEMORY (FAISS)
# ======================
@st.cache_resource
def load_embedder():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model, model.get_sentence_embedding_dimension()

embedder, DIM = load_embedder()
INDEX_FILE = "memory.index"
CHUNKS_FILE = "memory_chunks.pkl"

def load_memory():
    try:
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            index = faiss.read_index(INDEX_FILE)
            with open(CHUNKS_FILE, "rb") as f:
                chunks = pickle.load(f)
            return index, chunks
    except:
        pass
    return faiss.IndexFlatIP(DIM), []

if "index" not in st.session_state:
    st.session_state.index, st.session_state.doc_chunks = load_memory()

def add_to_memory(text):
    chunks = text.split("\n")
    vecs = embedder.encode(chunks, normalize_embeddings=True)
    st.session_state.index.add(vecs.astype("float32"))
    st.session_state.doc_chunks.extend(chunks)

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
# SMART PROJECT GENERATOR
# ======================
def create_project_zip(idea, tech):
    temp_dir = tempfile.mkdtemp()
    project_dir = os.path.join(temp_dir, "project")
    os.makedirs(project_dir, exist_ok=True)

    if tech == "Python - FastAPI":
        os.makedirs(os.path.join(project_dir, "app"), exist_ok=True)
        code = call_ai("Generate FastAPI main.py.", idea)
        with open(os.path.join(project_dir, "app", "main.py"), "w") as f:
            f.write(code)
        with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
            f.write("fastapi\nuvicorn\n")

    elif tech == "Python - Flask":
        code = call_ai("Generate Flask app.py.", idea)
        with open(os.path.join(project_dir, "app.py"), "w") as f:
            f.write(code)
        with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
            f.write("flask\n")

    elif tech == "Python - Streamlit":
        code = call_ai("Generate Streamlit app.py.", idea)
        with open(os.path.join(project_dir, "app.py"), "w") as f:
            f.write(code)
        with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
            f.write("streamlit\n")

    elif tech == "Node.js - Express":
        code = call_ai("Generate Node.js Express app.js.", idea)
        with open(os.path.join(project_dir, "app.js"), "w") as f:
            f.write(code)
        with open(os.path.join(project_dir, "package.json"), "w") as f:
            f.write("""{
  "name": "ai-project",
  "version": "1.0.0",
  "main": "app.js",
  "dependencies": {
    "express": "^4.18.2"
  }
}""")

    readme = call_ai("Create a README with run instructions.", idea)
    with open(os.path.join(project_dir, "README.md"), "w") as f:
        f.write(readme)

    zip_path = os.path.join(temp_dir, "project.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(project_dir):
            for file in files:
                fp = os.path.join(root, file)
                zipf.write(fp, os.path.relpath(fp, project_dir))

    return zip_path

# ======================
# UI
# ======================
st.set_page_config("Veera Enterprise AI", layout="wide")
st.title("🚀 Veera Enterprise AI Platform")

tabs = st.tabs([
    "💬 AI Chat",
    "📄 Document Memory",
    "🎤 Voice Reply",
    "⚙️ Automation",
    "🏗 Project Architecture",
    "🧪 Code Lab"
])

# AI CHAT
with tabs[0]:
    q = st.chat_input("Ask anything...")
    if q:
        ans = call_ai("General assistant.", q)
        st.write(ans)

# MEMORY
with tabs[1]:
    f = st.file_uploader("Upload PDF", type="pdf")
    if f:
        text = "".join(
            p.extract_text() for p in PdfReader(f).pages if p.extract_text()
        )
        add_to_memory(text)
        st.success("Document stored")

# VOICE
with tabs[2]:
    voice_q = st.text_input("Ask for voice reply")
    if voice_q:
        ans = call_ai("Voice assistant.", voice_q)
        st.write(ans)
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
        result = agent_workflow(task)
        st.write(result)

# ARCHITECTURE
with tabs[4]:
    idea = st.text_area("Describe entire project")
    if st.button("Generate Architecture"):
        result = agent_workflow(idea)
        st.write(result)

# CODE LAB
with tabs[5]:
    tech = st.selectbox(
        "Select Technology",
        [
            "Python - FastAPI",
            "Python - Flask",
            "Python - Streamlit",
            "Node.js - Express"
        ]
    )

    idea = st.text_area("Describe project for code generation")

    if st.button("Generate Project Code"):
        zip_path = create_project_zip(idea, tech)
        with open(zip_path, "rb") as f:
            st.download_button(
                "⬇️ Download Project ZIP",
                data=f,
                file_name="project.zip",
                mime="application/zip"
            )
