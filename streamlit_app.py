import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

st.title("Reconocimiento de Entidades Nombradas (NER) con XLM-RoBERTa")
st.write("Este chatbot identifica entidades como nombres, organizaciones, ubicaciones, entre otras, en un texto.")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    return pipeline("ner", model=model, tokenizer=tokenizer)

classifier = load_model()

user_input = st.text_area("Introduce tu texto aquí:", value="", placeholder="Escribe algo para analizar...")
if st.button("Analizar entidades"):
    if user_input.strip():
        with st.spinner("Procesando..."):
            results = classifier(user_input)
            st.subheader("Entidades identificadas:")
            for entity in results:
                st.write(f"- **Texto**: `{entity['word']}`")
                st.write(f"  - **Etiqueta**: `{entity['entity']}`")
                st.write(f"  - **Confianza**: `{entity['score']:.2f}`")
                st.write("---")
    else:
        st.warning("Por favor, introduce algún texto para analizar.")
