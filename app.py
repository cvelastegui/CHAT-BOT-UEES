import streamlit as st
from src.chatbot import get_bot_response

st.set_page_config(page_title="HERO Smart Advisor", page_icon="🏍️", layout="centered")

st.title("🏍️ HERO Smart Advisor")
st.caption("ChatBot inteligente con NLP, Deep Learning y motor de reglas")

with st.expander("¿Qué hace este prototipo?"):
    st.write(
        "Clasifica la intención del usuario mediante una red neuronal con Embedding y capas densas. "
        "Después aplica reglas de negocio para recomendar modelos u orientar la conversación."
    )

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "¡Hola! Soy HERO Smart Advisor. Pregúntame por modelos, precios, crédito, economía, trabajo, ciudad o aventura."
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Escribe tu consulta...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            result = get_bot_response(prompt)
        st.markdown(result["response"])
        st.caption(f"Intención detectada: {result['intent']} | Confianza: {result['confidence']:.1%}")

    st.session_state.messages.append({"role": "assistant", "content": result["response"]})

st.divider()
st.caption("Proyecto académico UEES · Inteligencia Artificial aplicada a los negocios")
