import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to allow importing 'core' and 'utils'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.perceptron_model import SimplePerceptron
from utils import get_numeric_columns, validate_binary_target

st.set_page_config(page_title="Perceptron", layout="wide")

st.markdown("## Perceptron Binary Classifier")
st.write("Upload a CSV to train a binary classifier. The model will identify original feature names, allow you to train, and provide a predict interface.")

st.markdown("### 1. Upload CSV Data")
uploaded_file = st.file_uploader("Upload dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Display the column names dynamically from the raw data
        st.write("Dataset loaded! Here are your original data features:")
        st.dataframe(df.head(), use_container_width=True)
        
        cols = get_numeric_columns(df)
        if not cols:
            st.error("No numeric columns available in the CSV to train on.")
            st.stop()
            
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select Target Variable (y)", options=cols, index=len(cols)-1)
        with col2:
            # Emphasize original data features
            feature_cols = st.multiselect("Select Features for training (X)", options=[c for c in cols if c != target_col], default=[c for c in cols if c != target_col])
        
        if not feature_cols:
            st.warning("Please select at least one feature.")
            st.stop()
            
        X = df[feature_cols].values
        y_raw = df[target_col].values
        
        is_binary, unique_targets, y = validate_binary_target(y_raw)
        if not is_binary:
            st.error("Perceptron only supports Binary targets (2 classes).")
            st.stop()
            
        st.markdown("### 2. Configure & Train")
        lr = st.number_input("Learning Rate", value=0.1, step=0.01)
        epochs = st.slider("Epochs", 1, 500, 50)
        
        # Train Button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training..."):
                model = SimplePerceptron(input_size=X.shape[1], lr=lr, epochs=epochs)
                errors, accuracies = model.fit(X, y)
                
                # Save into state for prediction
                st.session_state['p_model'] = model
                st.session_state['p_features'] = feature_cols
                st.session_state['p_targets'] = unique_targets
                
            st.success("Training Complete!")
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(errors, marker='o')
            ax[0].set_title('Errors over Epochs')
            ax[0].set_xlabel('Epoch')
            
            ax[1].plot(accuracies, color='green', marker='s')
            ax[1].set_title('Accuracy')
            ax[1].set_ylim(0, 1.05)
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error parsing dataframe: {e}")

st.markdown("---")
# Predict Button / Section
st.markdown("### 3. Predict")

if 'p_model' in st.session_state:
    st.write("Enter values for your **original features** to predict the class:")
    model = st.session_state['p_model']
    features = st.session_state['p_features']
    targets = st.session_state['p_targets']
    
    input_vals = []
    cols = st.columns(min(len(features), 4))
    for i, feature_name in enumerate(features):
        c_idx = i % 4
        # Shows exact fearue name from uploaded CSV
        val = cols[c_idx].number_input(f"{feature_name}", value=0.0)
        input_vals.append(val)
        
    if st.button("🔮 Predict", type="primary"):
        input_arr = np.array(input_vals)
        pred_bin = model.predict(input_arr)[0]
        
        if len(targets) == 2 and not set(targets).issubset({0, 1}):
            final_class = targets[0] if pred_bin == 0 else targets[1]
        else:
            final_class = pred_bin
            
        st.success(f"**Predicted Class:** {final_class}")
else:
    st.info("Upload CSV and Train a model first to predict new data.")
