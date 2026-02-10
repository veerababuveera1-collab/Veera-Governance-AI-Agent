import streamlit as st
import requests, os, time, faiss, numpy as np, logging, pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components
import zipfile
import tempfile
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

# ======================
# LOAD ENV
# ======================
load_dotenv()

# ======================
# LOGGING
# ======================
logging.basicConfig(level=logging.INFO)

# ======================
# AUTH (env + fallback)
# ======================
APP_USER = os.getenv("APP_USER", "veera")
APP_PASS = os.getenv("APP_PASS", "ai2026")

def login():
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == APP_USER and p == APP_PASS:
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid credentials")

if "user" not in st.session_state:
    login()
    st.stop()

# ======================
# AI CONFIG (Groq + Gemini)
# ======================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY and not GEMINI_API_KEY:
    st.error("No AI keys found. Add GROQ_API_KEY or GEMINI_API_KEY.")
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
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ======================
# PERSISTENT VECTOR MEMORY
# ======================
DIM = 384
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
        logging.warning("Memory load failed, creating new memory.")
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
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))
    return chunks

def add_to_memory(text):
    chunks = smart_chunks(text)
    vecs = embedder.encode(chunks, normalize_embeddings=True)
    st.session_state.index.add(vecs.astype("float32"))
    st.session_state.doc_chunks.extend(chunks)
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
# AI CALL (Groq + Gemini fallback)
# ======================
def call_ai(system, user, retries=2):
    prompt = f"{system}\n\n{user[:6000]}"

    # ---- GROQ FIRST ----
    if groq_client:
        for _ in range(retries):
            try:
                resp = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user[:6000]}
                    ]
                )
                return resp.choices[0].message.content
            except Exception as e:
                logging.warning(f"Groq failed: {e}")
                time.sleep(1)

    # ---- GEMINI FALLBACK ----
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
# 12-AGENT ENTERPRISE BRAIN
# ======================
AGENTS = {
    "planner": "Break the task into modules",
    "researcher": "Provide domain insights",
    "architect": "Design architecture",
    "tech_stack": "Select tech stack",
    "backend": "Design backend APIs",
    "frontend": "Design frontend UI",
    "database": "Design database schema",
    "devops": "Provide deployment plan",
    "qa": "Create test strategy",
    "security": "Identify security risks",
    "performance": "Suggest optimizations",
    "documentation": "Generate final summary"
}

def agent_workflow(q, context):
    outputs = {}
    context = context[-4000:]

    progress = st.progress(0)
    step = 0
    total = len(AGENTS)

    for agent, role in AGENTS.items():
        logging.info(f"Running agent: {agent}")
        prompt = context + "\n" + q
        result = call_ai(role, prompt)
        outputs[agent] = result
        context += f"\n\n{agent.upper()}:\n{result}"

        step += 1
        progress.progress(step / total)
        time.sleep(0.2)

    merged = "\n\n".join(
        f"{name.upper()}:\n{outputs[name]}"
        for name in outputs
    )

    return call_ai(
        "Combine into a single enterprise-grade response",
        merged
    )

# ======================
# PROJECT GENERATOR
# ======================
def generate_project_code(idea):
    return call_ai("Generate a simple working Streamlit app.", idea)

def generate_readme(idea):
    return call_ai("Create a README for this project.", idea)

def create_project_zip(app_code, readme_text):
    temp_dir = tempfile.mkdtemp()
    project_path = os.path.join(temp_dir, "project")
    os.makedirs(project_path, exist_ok=True)

    with open(os.path.join(project_path, "app.py"), "w") as f:
        f.write(app_code)

    with open(os.path.join(project_path, "README.md"), "w") as f:
        f.write(readme_text)

    with open(os.path.join(project_path, "requirements.txt"), "w") as f:
        f.write("streamlit\nrequests\n")

    zip_path = os.path.join(temp_dir, "project.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(project_path):
            for file in files:
                fp = os.path.join(root, file)
                zipf.write(fp, os.path.relpath(fp, project_path))

    return zip_path

# ======================
# VOICE OUTPUT
# ======================
def speak_browser(text):
    components.html(f"""
    <script>
    const msg = new SpeechSynthesisUtterance({repr(text)});
    window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

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
    "🧠 Project Generator"
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
        speak_browser(ans)

# AUTOMATION
with tabs[3]:
    task = st.text_area("Describe task")
    if st.button("Run Automation"):
        mem = search_memory(task)
        result = agent_workflow(task, mem)
        st.markdown(result)

# PROJECT GENERATOR
with tabs[4]:
    idea = st.text_area("Describe project")
    if st.button("Generate Project"):
        if idea.strip():
            app_code = generate_project_code(idea)
            readme = generate_readme(idea)
            zip_path = create_project_zip(app_code, readme)

            st.code(app_code, language="python")

            with open(zip_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Project ZIP",
                    data=f,
                    file_name="project.zip",
                    mime="application/zip"
                )

# LOGOUT
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
