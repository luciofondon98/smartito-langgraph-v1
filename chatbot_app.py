import streamlit as st
from main import graph, FlowState

st.set_page_config(page_title="SMARTito Chatbot", page_icon="✈️")
st.title("SMARTito Chatbot ✈️")
st.markdown("Interactúa con tu API de métricas aéreas usando lenguaje natural.")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_input("Haz tu pregunta:", key="user_input")

if st.button("Enviar") and user_input:
    state = FlowState(
        user_question=user_input,
        technical_query="",
        raw_data="",
        final_answer=""
    )
    result = graph.invoke(state)
    st.session_state["history"].append(("user", user_input))
    st.session_state["history"].append(("bot", result["final_answer"]))
    st.rerun()

for role, msg in reversed(st.session_state["history"]):
    if role == "user":
        st.markdown(f"**Tú:** {msg}")
    else:
        st.markdown(f"**SMARTito:** {msg}") 