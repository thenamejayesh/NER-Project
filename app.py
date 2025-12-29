import streamlit as st
import spacy
from spacy import displacy

st.set_page_config(
    page_title="NER App",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

# Loading model
with st.spinner("Loading NLP model... Please wait ⏳"):
    nlp = load_nlp()

st.title("🧠 Named Entity Recognition App")
st.markdown("Extract **Person, Location, Organization** and more from text.")

text = st.text_area(
    "✍️ Enter your text below:",
    "Virat Kohli was born in Delhi and plays cricket for India",
    height=150
)

if st.button("🔍 Analyze Text"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        doc = nlp(text)

        st.subheader("📌 Detected Entities")

        if doc.ents:
            for ent in doc.ents:
                st.success(f"Entity: {ent.text} | Label: {ent.label_} ({spacy.explain(ent.label_)})")
        else:
            st.info("No entities found.")

        st.subheader("📊 Visualization")
        html = displacy.render(doc, style="ent")
        st.markdown(html, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with ❤️ using SpaCy & Streamlit")

