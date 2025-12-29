
import streamlit as st
import spacy
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Deep Learning NER App",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_trf")  # Transformer-based deep learning model

nlp = load_model()

# ---------------- UI HEADER ----------------
st.markdown("""
<h1 style="text-align:center; color:#4CAF50;">Deep Learning Named Entity Recognition</h1>
<p style="text-align:center; font-size:18px;">
Powered by <b>SpaCy Transformer Model (en_core_web_trf)</b>
</p>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------
text = st.text_area(
    "✍️ Enter your text:",
    "Virat Kohli was born in Delhi and plays cricket for India.",
    height=150
)

# ---------------- BUTTON ----------------
if st.button("🔍 Analyze Text"):
    if text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        doc = nlp(text)

        if len(doc.ents) == 0:
            st.info("No named entities found.")
        else:
            # Create table
            data = []
            for ent in doc.ents:
                data.append({
                    "Entity": ent.text,
                    "Label": ent.label_,
                    "Meaning": spacy.explain(ent.label_)
                })

            df = pd.DataFrame(data)

            st.success("Named Entities Detected Successfully ✅")
            st.dataframe(df, use_container_width=True)

            # Highlighted output
            st.markdown("### 🖍 Highlighted Entities")
            highlighted_text = ""
            for ent in doc.ents:
                highlighted_text += f"""
                <span style="background-color:#ffeaa7;
                             padding:6px;
                             margin:4px;
                             border-radius:6px;
                             display:inline-block;">
                    {ent.text} <b>({ent.label_})</b>
                </span>
                """
            st.markdown(highlighted_text, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>🚀 Built with SpaCy Deep Learning & Streamlit</center>",
    unsafe_allow_html=True
)
