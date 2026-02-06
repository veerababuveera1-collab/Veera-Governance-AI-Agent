import gradio as gr
import requests
import os

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY environment variable")

SYSTEM_PROMPT = """
You are Veera's Governance AI Agent.
You think in systems, communicate clearly, and always propose next actions.
"""

MODEL_NAME = "meta-llama/llama-3-70b-instruct"

def agent_reply(messages):
    formatted = [{"role": "system", "content": SYSTEM_PROMPT}]
    formatted.extend(messages)

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={"model": MODEL_NAME, "messages": formatted},
            timeout=60
        )

        data = response.json()
        if "error" in data:
            return {"role": "assistant", "content": f"Error: {data['error']['message']}"}

        return data["choices"][0]["message"]

    except Exception as e:
        return {"role": "assistant", "content": f"Exception: {str(e)}"}


custom_css = """
.chatbot .message.user {
    background: linear-gradient(135deg, #0ea5e9, #22c55e);
    color: #0f172a;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 14px;
}

.chatbot .message.assistant {
    background: rgba(15,23,42,0.9);
    border-radius: 18px 18px 18px 4px;
    border: 1px solid rgba(148,163,184,0.4);
    padding: 10px 14px;
    color: #e5e7eb;
}

body {
    background: radial-gradient(circle at top, #0b1220 0, #020617 45%, #000000 100%);
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(
        """
        <div style="text-align:center; padding: 18px 0 6px;">
            <div style="font-size: 1.4rem; font-weight: 700; color: #e5e7eb;">
                Veera's Governance AI Agent
            </div>
            <div style="font-size: 0.9rem; color: #9ca3af; margin-top: 4px;">
                Llama‑3 70B · OpenRouter · Governance · Architecture · Next Actions
            </div>
        </div>
        """
    )

    chatbot = gr.Chatbot(type="messages", height=500, elem_classes=["chatbot"])

    with gr.Row():
        textbox = gr.Textbox(
            placeholder="Ask about governance, defects, KPIs, architecture, or stories...",
            lines=3,
            scale=8
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", variant="secondary", scale=1)

    def user_send(message, history):
        history = history + [{"role": "user", "content": message}]
        return "", history

    def bot_reply(history):
        reply = agent_reply(history)
        history = history + [reply]
        return history

    send_btn.click(user_send, [textbox, chatbot], [textbox, chatbot]).then(
        bot_reply, chatbot, chatbot
    )

    clear_btn.click(lambda: [], None, chatbot)

Demo.launch()
