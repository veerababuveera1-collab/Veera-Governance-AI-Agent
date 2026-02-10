import streamlit as st
from groq import Groq

st.title("Groq Test")

if st.button("Test Groq"):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello"}]
        )

        st.success(response.choices[0].message.content)

    except Exception as e:
        st.error(f"Groq error: {str(e)}")
