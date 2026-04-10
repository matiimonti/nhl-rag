import os
import requests
import streamlit as st

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.set_page_config(page_title="NHL RAG", page_icon="🏒")
st.title("🏒 NHL RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something about the NHL:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{FASTAPI_URL}/query",
                    json={"question": prompt, "top_k": 5},
                    timeout=30,
                )
                if resp.status_code == 501:
                    answer = "_(stub) Query pipeline not implemented yet._"
                else:
                    data = resp.json()
                    answer = data.get("answer", "No answer returned.")
            except requests.exceptions.ConnectionError:
                answer = "Could not reach the API. Is FastAPI running?"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
