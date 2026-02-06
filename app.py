import gradio as gr
import requests
import os

# ==============================
# CONFIG
# ==============================

os.environ["OPENROUTER_API_KEY"] = "YOUR_API_KEY_HERE"
API_KEY = os.environ.get("OPENROUTER_API_KEY")

MODEL_NAME = "meta-llama/llama-3-70b-instruct"

SYSTEM_PROMPT = """
You are Veera's Governance AI Agent.
You think in systems, explain clearly, and always propose next actions.
"""

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ==============================
# LLM CALL
# ==============================

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
        res = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        data = res.json()

        if "error" in data:
            return {"role": "assistant", "content": f"❌ {data['error']['message']}"}

        return data["choices"][0]["message"]

    except Exception as e:
        return {"role": "assistant", "content": f"⚠️ {str(e)}"}


# ==============================
# UI STYLE
# ==============================

CUSTOM_CSS = """
.chatbot .message.user {
    background: linear-gradient(135deg,#0ea5e9,#22c55e);
    color:#0f172a;
    border-radius:18px 18px 4px 18px;
    padding:10px 14px;
}

.chatbot .message.assistant {
    background:#0f172a;
    color:#e5e7eb;
    border-radius:18px 18px 18px 4px;
    border:1px solid #334155;
    padding:10px 14px;
}

body {
    background: radial-gradient(circle at top,#0b1220,#000);
}
"""


# ==============================
# APP
# ==============================

with gr.Blocks(css=CUSTOM_CSS) as demo:

    gr.Markdown("""
    <div style="text-align:center;padding:16px">
        <h2 style="color:#e5e7eb">Veera's Governance AI Agent</h2>
        <p style="color:#9ca3af">Llama-3 70B · OpenRouter · Intelligent Systems</p>
    </div>
    """)

    chatbot = gr.Chatbot(type="messages", height=500, elem_classes=["chatbot"])

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Ask about governance, architecture, KPIs, defects...",
            lines=3,
            scale=8
        )
        send_btn = gr.Button("Send", scale=1)
        clear_btn = gr.Button("Clear", scale=1)


    def send_message(message, history):
        history = history + [{"role": "user", "content": message}]
        return "", history


    def get_reply(history):
        reply = call_agent(history)
        history = history + [reply]
        return history


    send_btn.click(
        send_message,
        [user_input, chatbot],
        [user_input, chatbot]
    ).then(
        get_reply,
        chatbot,
        chatbot
    )

    clear_btn.click(lambda: [], None, chatbot)


# ==============================
# RUN
# ==============================

demo.launch(share=True)
