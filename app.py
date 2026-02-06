import streamlit as st
import requests
import os

# ============================
# CONFIG
# ============================

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    st.error("❌ OPENROUTER_API_KEY not found. Add it in GitHub/Streamlit secrets.")
    st.stop()

MODEL_NAME = "meta-llama/llama-3-70b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """
You are Veera's Governance AI Agent.
You think in systems, explain clearly, and always propose next actions.
"""


# ============================
# LLM CALL
# ============================

def call_agent(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        data = r.json()

        if "error" in data:
            return f"❌ {data['error']['message']}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠️ {str(e)}"


# ============================
# STREAMLIT UI
# ============================

st.set_page_config(page_title="Veera Governance AI", layout="centered")

st.title("🧠 Veera's Governance AI Agent")
st.caption("OpenRouter · Llama-3 70B · Systems Thinking AI")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Show chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Ask about governance, KPIs, defects, architecture...")

if prompt:
    # Add user message
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = call_agent(st.session_state.chat)
            st.markdown(reply)

    st.session_state.chat.append({"role": "assistant", "content": reply})

# Clear button
if st.button("🧹 Clear chat"):
    st.session_state.chat = []
    st.rerun()
