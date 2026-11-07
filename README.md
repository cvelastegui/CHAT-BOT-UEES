# HERO Smart Advisor

Proyecto académico de Inteligencia Artificial aplicada a los negocios.

## Descripción

HERO Smart Advisor es un ChatBot que combina:

- Procesamiento de lenguaje natural
- Red neuronal con capa Embedding
- Clasificación de intenciones
- Motor de reglas
- Base de conocimiento
- Interfaz conversacional en Streamlit

## Arquitectura

Usuario → Streamlit → NLP → Embedding + MLP → Intención → Motor de reglas → Respuesta

## Instalación local

```bash
git clone TU_URL_DEL_REPOSITORIO
cd hero-smart-advisor
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Instala dependencias:

```bash
pip install -r requirements.txt
```

## Entrenar el modelo

```bash
python train.py
```

El entrenamiento genera:

- `models/intent_model.keras`
- `models/tokenizer.pkl`
- `models/label_encoder.pkl`
- `models/metrics.json`

## Ejecutar el ChatBot

```bash
streamlit run app.py
```

La aplicación también intenta entrenar automáticamente el modelo si los archivos del modelo todavía no existen.

## Ejecutar las 20 pruebas

Desde la raíz del proyecto:

```bash
python -m tests.test_chatbot
```

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub.
2. Entra a Streamlit Community Cloud.
3. Selecciona tu repositorio.
4. Como archivo principal usa `app.py`.
5. Despliega.

## Nota académica

Los resultados de accuracy, precision, recall y F1 deben tomarse del archivo `models/metrics.json` después de entrenar el proyecto. No deben inventarse métricas.
