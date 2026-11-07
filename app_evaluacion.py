import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.chatbot import get_bot_response
from tests.test_chatbot import TESTS

BASE = Path(__file__).resolve().parent
METRICS_PATH = BASE / "models" / "metrics.json"

st.set_page_config(page_title="HERO Smart Advisor", page_icon="🏍️", layout="centered")

st.title("🏍️ HERO Smart Advisor")
st.caption("ChatBot inteligente con NLP, Deep Learning y motor de reglas")

tab_chat, tab_eval = st.tabs(["💬 ChatBot", "📊 Evaluación"])

with tab_chat:
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

with tab_eval:
    st.subheader("Evaluación del modelo")
    st.write(
        "Esta sección ejecuta las 20 interacciones de prueba exigidas para documentar el desempeño funcional del ChatBot."
    )

    if st.button("▶ Ejecutar 20 pruebas", type="primary"):
        rows = []
        correct = 0
        progress = st.progress(0)

        for i, (text, expected) in enumerate(TESTS, start=1):
            result = get_bot_response(text)
            pred = result["intent"]
            ok = pred == expected
            correct += int(ok)
            rows.append({
                "#": i,
                "Entrada": text,
                "Esperada": expected,
                "Predicha": pred,
                "Confianza": round(result["confidence"] * 100, 1),
                "Resultado": "OK" if ok else "FALLO"
            })
            progress.progress(i / len(TESTS))

        df = pd.DataFrame(rows)
        success = correct / len(TESTS)
        st.session_state["eval_df"] = df
        st.session_state["eval_success"] = success

    if "eval_df" in st.session_state:
        success = st.session_state["eval_success"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Pruebas correctas", f"{int(success * len(TESTS))}/{len(TESTS)}")
        c2.metric("Tasa de éxito", f"{success:.1%}")
        c3.metric("Pruebas", len(TESTS))
        st.dataframe(st.session_state["eval_df"], use_container_width=True, hide_index=True)

        csv = st.session_state["eval_df"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar resultados CSV",
            data=csv,
            file_name="resultados_20_pruebas.csv",
            mime="text/csv"
        )

    st.divider()
    st.subheader("Métricas de clasificación NLP")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.1%}")
        m2.metric("Precision", f"{metrics.get('macro_precision', 0):.1%}")
        m3.metric("Recall", f"{metrics.get('macro_recall', 0):.1%}")
        m4.metric("F1 macro", f"{metrics.get('macro_f1', 0):.1%}")
        st.caption(
            f"Dataset: {metrics.get('dataset_size', 'N/D')} frases | "
            f"Épocas ejecutadas: {metrics.get('epochs_trained', 'N/D')}"
        )
    else:
        st.info("Ejecuta las 20 pruebas para inicializar/entrenar el modelo; luego las métricas aparecerán aquí.")

st.divider()
st.caption("Proyecto académico UEES · Inteligencia Artificial aplicada a los negocios")
