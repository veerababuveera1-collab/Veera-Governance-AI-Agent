import streamlit as st
import os, time, logging, zipfile, tempfile, hashlib
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

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

    return call_ai(
        """Combine into structured sections:
1. Overview
2. Modules
3. Tech Stack
4. Architecture
5. Database
6. APIs
7. Deployment
8. Security
9. Testing
10. Summary
""",
        merged
    )

# ======================
# PROJECT GENERATOR
# ======================
def generate_backend(idea):
    return call_ai("Generate backend code (Python/Flask or FastAPI).", idea)

def generate_frontend(idea):
    return call_ai("Generate simple frontend HTML interface.", idea)

def generate_database(idea):
    return call_ai("Generate database models.", idea)

def generate_readme(idea):
    return call_ai("Create a professional README.", idea)

def create_project_zip(idea):
    temp_dir = tempfile.mkdtemp()
    project_path = os.path.join(temp_dir, "project")
    os.makedirs(project_path, exist_ok=True)

    backend_dir = os.path.join(project_path, "backend")
    frontend_dir = os.path.join(project_path, "frontend")
    db_dir = os.path.join(project_path, "database")

    os.makedirs(backend_dir)
    os.makedirs(frontend_dir)
    os.makedirs(db_dir)

    backend = generate_backend(idea)
    frontend = generate_frontend(idea)
    database = generate_database(idea)
    readme = generate_readme(idea)

    with open(os.path.join(backend_dir, "app.py"), "w") as f:
        f.write(backend)

    with open(os.path.join(frontend_dir, "index.html"), "w") as f:
        f.write(frontend)

    with open(os.path.join(db_dir, "models.py"), "w") as f:
        f.write(database)

    with open(os.path.join(project_path, "requirements.txt"), "w") as f:
        f.write("fastapi\nflask\nstreamlit\n")

    with open(os.path.join(project_path, "README.md"), "w") as f:
        f.write(readme)

    zip_path = os.path.join(temp_dir, "project.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(project_path):
            for file in files:
                fp = os.path.join(root, file)
                zipf.write(fp, os.path.relpath(fp, project_path))

    return zip_path

# ======================
# UI
# ======================
st.set_page_config("Veera Enterprise AI", layout="wide")
st.title("🚀 Veera Enterprise AI Platform")

tabs = st.tabs([
    "🏗 Project Architecture",
    "🧪 Code Lab"
])

# TAB 1: ARCHITECTURE
with tabs[0]:
    st.subheader("Project Architecture Generator")
    idea = st.text_area("Describe entire project")

    if st.button("Generate Architecture"):
        if idea.strip():
            with st.spinner("Designing architecture..."):
                result = agent_workflow(idea)
            st.markdown(result)

# TAB 2: CODE LAB
with tabs[1]:
    st.subheader("Project Code Generator")
    idea = st.text_area("Describe project for code generation")

    if st.button("Generate Project Code"):
        if idea.strip():
            with st.spinner("Generating full project..."):
                zip_path = create_project_zip(idea)

            with open(zip_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Project ZIP",
                    data=f,
                    file_name="project.zip",
                    mime="application/zip"
                )
