import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.mlp_model import MLP
from utils import get_numeric_columns, validate_binary_target, standardize_data

st.set_page_config(page_title="Backpropagation", layout="wide")

st.markdown("## Multi-Layer Perceptron (Backprop)")
st.write("Train a neural network with one hidden layer on your CSV data. Handles non-linear relationships.")

st.markdown("### 1. Upload CSV Data")
uploaded_file = st.file_uploader("Upload dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset loaded! Raw Features:")
        st.dataframe(df.head(), use_container_width=True)
        
        cols = get_numeric_columns(df)
        if not cols: st.stop()
            
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select Target Variable (y)", options=cols, index=len(cols)-1)
        with col2:
            feature_cols = st.multiselect("Select Features for training (X)", options=[c for c in cols if c != target_col], default=[c for c in cols if c != target_col])
        
        if not feature_cols: st.stop()
            
        X_raw = df[feature_cols].values
        y_raw = df[target_col].values
        
        is_binary, unique_targets, y = validate_binary_target(y_raw)
        y = y.reshape(-1, 1) # Requires 2D column for MLP
        
        X, X_mean, X_std = standardize_data(X_raw)
            
        st.markdown("### 2. Configure & Train")
        c1, c2, c3 = st.columns(3)
        hidden_nodes = c1.number_input("Hidden Nodes", value=8, min_value=1)
        lr = c2.number_input("Learning Rate", value=0.1, step=0.01)
        epochs = c3.number_input("Epochs", value=1000, step=100)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training MLP..."):
                model = MLP(input_dim=X.shape[1], hidden_dim=hidden_nodes, output_dim=1, lr=lr)
                losses = []
                accs = []
                
                prog = st.progress(0)
                for i in range(int(epochs)):
                    model.forward(X)
                    loss = model.backward(X, y)
                    if i % max(1, epochs//100) == 0 or i == epochs - 1:
                        preds = model.predict(X)
                        acc = np.mean(preds == y)
                        losses.append(loss)
                        accs.append(acc)
                        prog.progress((i+1)/epochs)
                
                st.session_state['mlp_model'] = model
                st.session_state['mlp_features'] = feature_cols
                st.session_state['mlp_targets'] = unique_targets
                st.session_state['mlp_mean'] = X_mean
                st.session_state['mlp_std'] = X_std
                
            st.success("Training Complete!")
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(losses, color='red'); ax[0].set_title('Loss (MSE)')
            ax[1].plot(accs, color='green'); ax[1].set_title('Accuracy')
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error parsing dataframe: {e}")

st.markdown("---")
st.markdown("### 3. Predict")

if 'mlp_model' in st.session_state:
    st.write("Enter values for your **original features** to predict the class:")
    model = st.session_state['mlp_model']
    features = st.session_state['mlp_features']
    targets = st.session_state['mlp_targets']
    x_mean = st.session_state['mlp_mean']
    x_std = st.session_state['mlp_std']
    
    input_vals = []
    cols = st.columns(min(len(features), 4))
    for i, feature_name in enumerate(features):
        c_idx = i % 4
        val = cols[c_idx].number_input(f"{feature_name}", value=0.0)
        input_vals.append(val)
        
    if st.button("🔮 Predict", type="primary"):
        input_arr = np.array(input_vals).reshape(1, -1)
        scaled_input = (input_arr - x_mean) / x_std
        
        pred_prob = model.forward(scaled_input)[0][0]
        pred_bin = 1 if pred_prob >= 0.5 else 0
        
        if len(targets) == 2 and not set(targets).issubset({0, 1}):
            final_class = targets[0] if pred_bin == 0 else targets[1]
        else:
            final_class = pred_bin
            
        st.success(f"**Predicted Class:** {final_class} (Confidence: {pred_prob:.4f})")
else:
    st.info("Upload CSV and Train a model first to predict new data.")
