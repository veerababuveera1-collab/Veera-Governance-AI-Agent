import streamlit as st
import requests, os, time, faiss, numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components
from datetime import datetime

# ======================
# AUTH
# ======================

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

# ======================
# CONFIG
# ======================

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    st.error("Missing OPENROUTER_API_KEY")
    st.stop()

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3-70b-instruct"

# ======================
# SAFE LLM CALL
# ======================

def call_ai(messages, max_tokens=1000, retries=2):
    for _ in range(retries):
        try:
            r = requests.post(
                URL,
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

            if "error" in data:
                return f"⚠️ AI Error: {data['error'].get('message','Unknown')}"

            return "⚠️ Empty response"

        except Exception:
            time.sleep(1)

    return "❌ AI service unavailable"

# ======================
# AGENTS
# ======================

class BaseAgent:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt

    def run(self, user_input, context=""):
        return call_ai([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context + "\n" + user_input}
        ])

class ChatAgent(BaseAgent):
    def __init__(self):
        super().__init__("You are a helpful AI assistant.")

class AutomationAgent(BaseAgent):
    def __init__(self):
        super().__init__("You are an automation AI. Provide step-by-step execution.")

class CodeAgent(BaseAgent):
    def __init__(self):
        super().__init__("""
You are a senior software engineer.

You must generate a COMPLETE, SINGLE-FILE, FULLY RUNNABLE program.

STRICT RULES:
- Output ONLY code
- No explanations
- No partial classes
- No continuation
- No placeholders
- Include all imports
- Include all classes and functions
- Include main entry point
- Must run immediately
""")

class SecurityAgent(BaseAgent):
    def __init__(self):
        super().__init__("""
You are a senior security engineer.

Scan the code and detect:
- Hardcoded secrets
- SQL injection
- Unsafe eval/exec
- Insecure file handling
- Weak authentication logic

Output:
1. Issues
2. Risk level
3. Fix suggestions
""")

    def scan(self, code):
        return call_ai([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": code}
        ], max_tokens=600)

class AgentController:
    def __init__(self):
        self.chat = ChatAgent()
        self.automation = AutomationAgent()
        self.code = CodeAgent()
        self.security = SecurityAgent()

controller = AgentController()

# ======================
# MEMORY SYSTEM
# ======================

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

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
# SESSION STATE
# ======================

if "projects" not in st.session_state:
    st.session_state.projects = ["Default Project"]

if "activity" not in st.session_state:
    st.session_state.activity = []

if "chat" not in st.session_state:
    st.session_state.chat = []

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
# UI
# ======================

st.set_page_config("Veera Enterprise AI", layout="wide")
st.title("🚀 Veera Enterprise AI Platform")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Projects",
        "Code Lab",
        "Security",
        "Automation",
        "AI Chat",
        "Activity",
        "Billing"
    ]
)

# ======================
# DASHBOARD
# ======================

if menu == "Dashboard":
    st.subheader("Dashboard")
    st.metric("Projects", len(st.session_state.projects))
    st.metric("Activity Logs", len(st.session_state.activity))

    st.write("Recent Activity:")
    for a in st.session_state.activity[-5:]:
        st.write("•", a)

# ======================
# PROJECTS
# ======================

elif menu == "Projects":
    st.subheader("Projects")

    new_proj = st.text_input("New project name")
    if st.button("Create Project") and new_proj:
        st.session_state.projects.append(new_proj)
        st.success("Project created")

    for p in st.session_state.projects:
        st.write("•", p)

# ======================
# CODE LAB (CORE)
# ======================

elif menu == "Code Lab":
    st.subheader("💻 Code Lab")

    project = st.text_area(
        "Describe your system",
        placeholder="Build a hospital appointment system"
    )

    mode = st.selectbox(
        "Output type",
        ["Python Script", "Streamlit App", "FastAPI API"]
    )

    if st.button("Generate Executable Code"):
        if not project.strip():
            st.warning("Please describe a system.")
        else:
            base_prompt = controller.code.system_prompt

            if mode == "Streamlit App":
                base_prompt += "\nCreate a Streamlit app runnable with: streamlit run app.py"
            elif mode == "FastAPI API":
                base_prompt += "\nCreate a FastAPI app runnable with: uvicorn app:app --reload"
            else:
                base_prompt += "\nCreate a Python script runnable with: python app.py"

            initial_code = call_ai([
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": project}
            ], max_tokens=1800)

            fix_prompt = """
Review the following code and ensure:
- Fully executable
- No syntax errors
- All imports included
- Has valid entry point

Fix issues and output full code only.
"""

            final_code = call_ai([
                {"role": "system", "content": fix_prompt},
                {"role": "user", "content": initial_code}
            ], max_tokens=1800)

            st.success("Executable code generated")
            st.code(final_code, language="python")

            st.download_button(
                "Download app.py",
                final_code,
                file_name="app.py"
            )

            # Security scan
            report = controller.security.scan(final_code)
            st.subheader("Security Report")
            st.markdown(report)

            st.session_state.activity.append(
                f"{datetime.now().strftime('%H:%M:%S')} - Code generated"
            )

# ======================
# SECURITY
# ======================

elif menu == "Security":
    st.subheader("Security Center")
    st.write("Security scans appear after code generation.")

# ======================
# AUTOMATION
# ======================

elif menu == "Automation":
    task = st.text_area("Describe task")
    if st.button("Run Automation"):
        res = controller.automation.run(task)
        st.write(res)
        st.session_state.activity.append(
            f"{datetime.now().strftime('%H:%M:%S')} - Automation run"
        )

# ======================
# CHAT
# ======================

elif menu == "AI Chat":
    for m in st.session_state.chat:
        st.write(m)

    q = st.text_input("Ask something")
    if q:
        mem = search_memory(q)
        ans = controller.chat.run(q, mem)
        st.write(ans)
        st.session_state.chat.append(q)
        st.session_state.chat.append(ans)

# ======================
# ACTIVITY
# ======================

elif menu == "Activity":
    st.subheader("Activity Logs")
    for a in st.session_state.activity:
        st.write("•", a)

# ======================
# BILLING
# ======================

elif menu == "Billing":
    st.subheader("Billing")
    st.write("Plan: Starter")
    st.write("Usage: 20,000 tokens")

# ======================
# LOGOUT
# ======================

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()
