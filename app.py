from groq import Groq

if st.button("Test Groq"):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Hello"}]
        )
        st.success(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Groq error: {e}")
