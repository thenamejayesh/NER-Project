import streamlit as st
from transformers import pipeline
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Deep Learning NER App",
    page_icon="🧠",
    layout="centered"
)

# Load NER model (cached for performance)
@st.cache_resource
def load_ner_model():
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

ner_model = load_ner_model()

# UI Header
st.markdown(
    """
    <h1 style='text-align:center;color:#4CAF50;'>Deep Learning NER System</h1>
    <p style='text-align:center;'>Named Entity Recognition using BERT (Transformer Model)</p>
    """,
    unsafe_allow_html=True
)

# Input text
text = st.text_area(
    "✍️ Enter your text:",
    "Virat Kohli was born in Delhi and plays cricket for India."
)

# Button
if st.button("🔍 Extract Entities"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        results = ner_model(text)

        if not results:
            st.info("No named entities found.")
        else:
            data = []
            for ent in results:
                data.append({
                    "Entity": ent["word"],
                    "Label": ent["entity_group"],
                    "Confidence": round(ent["score"], 3)
                })

            df = pd.DataFrame(data)

            st.success("Entities Extracted Successfully ✅")
            st.dataframe(df, use_container_width=True)

            # Highlight Entities
            st.markdown("### 🖍 Highlighted Entities")
            highlighted = ""
            for ent in results:
                highlighted += f"""
                <span style="background-color:#ffd54f;
                             padding:4px;
                             border-radius:6px;
                             margin:3px;
                             display:inline-block;">
                {ent['word']} ({ent['entity_group']})
                </span>
                """
            st.markdown(highlighted, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<center>🚀 Built using Transformers & Streamlit | Deep Learning NLP App</center>",
    unsafe_allow_html=True
)
