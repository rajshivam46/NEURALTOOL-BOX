import streamlit as st

st.set_page_config(page_title="Neural Network Toolbox", layout="wide", page_icon="🧠")

st.markdown("# 🧠 Neural Network Toolbox")
st.markdown("### Welcome to the educational machine learning platform.")

st.write("Please select an algorithm from the sidebar to begin!")

st.info("""
**How to use modules:**
1. **Upload your CSV**: The toolbox reads the exact original features from your data.
2. **Train**: Click the train button to fit the network parameters.
3. **Predict**: Enter specific original feature values to get predictions directly from the trained network.
""")
