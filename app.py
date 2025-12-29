import streamlit as st
import spacy
import pandas as pd

# Page config
st.set_page_config(
    page_title="Named Entity Recognition (NER)",
    page_icon="🧠",
    layout="centered"
)

# Load NLP model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# Title & Description
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>Named Entity Recognition App</h1>
    <p style='text-align: center; font-size: 18px;'>
    Enter a sentence and identify entities like <b>Person, Location, Organization</b> etc.
    </p>
    """,
    unsafe_allow_html=True
)

# Input box
text = st.text_area(
    "✍️ Enter your text here:",
    "Virat Kohli was born in Delhi and plays cricket for India."
)

# Button
if st.button("🔍 Extract Entities"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        doc = nlp(text)

        if len(doc.ents) == 0:
            st.info("No entities found.")
        else:
            data = []
            for ent in doc.ents:
                data.append({
                    "Entity": ent.text,
                    "Label": ent.label_,
                    "Meaning": spacy.explain(ent.label_)
                })

            df = pd.DataFrame(data)

            st.success("Entities Extracted Successfully ✅")
            st.dataframe(df, use_container_width=True)

            # Highlighted text output
            st.markdown("### 🖍 Highlighted Text")
            html = ""
            for ent in doc.ents:
                html += f"<span style='background-color:#ffeaa7; padding:4px; border-radius:5px; margin-right:5px;'>{ent.text} ({ent.label_})</span> "
            st.markdown(html, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<center>🚀 Built with Streamlit & SpaCy | NLP Project</center>",
    unsafe_allow_html=True
)

