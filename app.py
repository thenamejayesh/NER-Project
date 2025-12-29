import streamlit as st
import spacy
from spacy import displacy

# Load NLP model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# Page configuration
st.set_page_config(
    page_title="Named Entity Recognition (NER)",
    page_icon="🧠",
    layout="centered"
)

# App Title
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🧠 Named Entity Recognition App</h1>
    <p style='text-align: center;'>Extract entities like Person, Location, Organization from text</p>
    """,
    unsafe_allow_html=True
)

# Input text
text = st.text_area(
    "✍️ Enter your sentence below:",
    "Virat Kohli was born in Delhi and plays cricket for India",
    height=150
)

# Button
if st.button("🔍 Analyze Text"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        doc = nlp(text)

        st.subheader("📌 Extracted Entities")

        if doc.ents:
            for ent in doc.ents:
                st.success(f"**Entity:** {ent.text}  |  **Label:** {ent.label_} ({spacy.explain(ent.label_)})")
        else:
            st.info("No entities found.")

        # Visualization
        st.subheader("📊 Entity Visualization")
        html = displacy.render(doc, style="ent", jupyter=False)
        st.markdown(html, unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align:center;'>🚀 Built using SpaCy & Streamlit</p>
    """,
    unsafe_allow_html=True
)
