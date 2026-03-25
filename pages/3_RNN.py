import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.rnn_model import SimpleRNN
from utils import get_numeric_columns

st.set_page_config(page_title="RNN Forecast", layout="wide")

st.markdown("## Recurrent Neural Network (RNN)")
st.write("Upload a time-series or sequential CSV to forecast the next value in the sequence.")

st.markdown("### 1. Upload CSV Data")
uploaded_file = st.file_uploader("Upload continuous sequential dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)
        
        cols = get_numeric_columns(df)
        if not cols: st.stop()
            
        target_col = st.selectbox("Select Target Variable to Sequence", options=cols)
        seq_length = st.slider("Window Length (Historical Steps to view)", 2, 20, 5)
        
        series = df[target_col].values
        series_mean = np.mean(series)
        series_std = np.std(series) + 1e-8
        series_norm = (series - series_mean) / series_std
        
        X = []
        Y = []
        for i in range(len(series_norm) - seq_length):
            X.append(series_norm[i:i+seq_length])
            Y.append(series_norm[i+seq_length])
            
        X_train = np.array(X)[..., np.newaxis]
        y_train = np.array(Y)
            
        st.markdown("### 2. Configure & Train")
        hidden_dim = st.number_input("Hidden Dimension", value=8, min_value=1)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training RNN via BPTT..."):
                model = SimpleRNN(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
                
                losses = []
                prog = st.progress(0)
                epochs = 100
                
                for epoch in range(epochs):
                    epoch_loss = 0
                    idx = np.random.choice(len(X_train), size=min(10, len(X_train)), replace=False)
                    for i in idx:
                        loss = model.train_step(X_train[i], y_train[i])
                        epoch_loss += loss
                    
                    losses.append(epoch_loss / len(idx))
                    prog.progress((epoch+1)/epochs)
                
                st.session_state['rnn_model'] = model
                st.session_state['rnn_seq_len'] = seq_length
                st.session_state['rnn_mean'] = series_mean
                st.session_state['rnn_std'] = series_std
                
            st.success("Training Complete!")
            st.line_chart(losses)
            
    except Exception as e:
        st.error(f"Error parsing dataframe: {e}")

st.markdown("---")
st.markdown("### 3. Predict")

if 'rnn_model' in st.session_state:
    st.write("Enter the preceding sequence values to forecast the next step:")
    model = st.session_state['rnn_model']
    sl = st.session_state['rnn_seq_len']
    s_m = st.session_state['rnn_mean']
    s_s = st.session_state['rnn_std']
    
    input_vals = []
    cols = st.columns(min(sl, 5))
    for i in range(sl):
        c_idx = i % 5
        val = cols[c_idx].number_input(f"Step T-{sl-i}", value=0.0)
        input_vals.append(val)
        
    if st.button("🔮 Forecast Next Phase", type="primary"):
        seq_input = np.array(input_vals).reshape(-1, 1)
        scaled_input = (seq_input - s_m) / s_s
        
        pred_scaled = model.predict(scaled_input)[0][0]
        pred_actual = (pred_scaled * s_s) + s_m
            
        st.success(f"**Forecasted Value:** {pred_actual:.4f}")
else:
    st.info("Upload CSV and Train a model first to predict new data.")
