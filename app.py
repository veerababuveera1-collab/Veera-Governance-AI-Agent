import streamlit as st
import os
from groq import Groq

st.title("Groq Test")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

if st.button("Test Groq"):
    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": "Hello"}]
    )
    st.write(resp.choices[0].message.content)
