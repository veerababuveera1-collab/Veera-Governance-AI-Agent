import streamlit as st
from groq import Groq

st.title("Groq Test")

if st.button("Test Groq"):
    st.write("Button clicked")

    try:
        api_key = st.secrets["GROQ_API_KEY"]
        st.write("API key loaded")

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Hello"}]
        )

        st.success(response.choices[0].message.content)

    except Exception as e:
        st.error(f"Groq error: {str(e)}")
