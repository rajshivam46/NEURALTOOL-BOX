import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loss Functions", layout="wide")

st.markdown("## Continuous Loss Analysis")
st.write("Visually comprehend how optimization goals shift depending on the choice of loss metric.")

loss_types = ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "Binary Cross-Entropy (BCE)"]
selected_loss = st.selectbox("Select Optimization Metric", loss_types)

st.markdown("---")

if selected_loss == "Mean Squared Error (MSE)":
    st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
    
    with st.sidebar:
        st.markdown("### Evaluation Parameters")
        y_true = st.number_input("Ground Truth (y)", value=0.0)
        
    y_pred = np.linspace(-10, 10, 200)
    loss = np.square(y_true - y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_pred, loss, label='MSE Gradient Path', color='blue', linewidth=2)
    ax.axvline(x=y_true, color='red', linestyle='--', label=f'Target ({y_true})')
    ax.set_xlabel('Network Output ($\hat{y}$)')
    ax.set_ylabel('Loss Penalty')
    ax.legend()
    st.pyplot(fig)
    
elif selected_loss == "Mean Absolute Error (MAE)":
    st.latex(r"MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|")
    
    with st.sidebar:
        st.markdown("### Evaluation Parameters")
        y_true = st.number_input("Ground Truth (y)", value=0.0)
        
    y_pred = np.linspace(-10, 10, 200)
    loss = np.abs(y_true - y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_pred, loss, label='MAE Gradient Path', color='green', linewidth=2)
    ax.axvline(x=y_true, color='red', linestyle='--', label=f'Target ({y_true})')
    ax.set_xlabel('Network Output ($\hat{y}$)')
    ax.set_ylabel('Loss Penalty')
    ax.legend()
    st.pyplot(fig)
    
elif selected_loss == "Binary Cross-Entropy (BCE)":
    st.latex(r"BCE = - \frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]")
    
    with st.sidebar:
        st.markdown("### Evaluation Parameters")
        y_true = st.radio("Ground Truth Class (y)", [0, 1], index=1)
        
    y_pred = np.linspace(0.001, 0.999, 200)
    if y_true == 1:
        loss = -np.log(y_pred)
    else:
        loss = -np.log(1 - y_pred)
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_pred, loss, label=f'BCE Loss (y={y_true})', color='orange', linewidth=2)
    ax.set_xlabel('Predicted Probability ($\hat{y}$)')
    ax.set_ylabel('Logarithmic Penalty')
    ax.legend()
    st.pyplot(fig)
