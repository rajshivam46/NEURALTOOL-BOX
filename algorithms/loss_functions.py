import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def app():
    st.markdown("## Continuous Loss Analysis")
    st.write("Visually comprehend how optimization goals shift depending on the choice of loss metric.")
    
    loss_types = ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "Binary Cross-Entropy (BCE)"]
    selected_loss = st.selectbox("Select Optimization Metric", loss_types)
    
    st.markdown("---")
    
    if selected_loss == "Mean Squared Error (MSE)":
        st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
        st.info("**Application:** Regression. Heavily penalizes outliers due to the squared term.")
        
        with st.sidebar:
            st.markdown("### Evaluation Parameters")
            y_true = st.number_input("Ground Truth (y)", value=0.0)
            
        y_pred = np.linspace(-10, 10, 200)
        loss = np.square(y_true - y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_pred, loss, label='MSE Gradient Path', color='#4f46e5', linewidth=3)
        ax.axvline(x=y_true, color='#ef4444', linestyle='--', label=f'Target ({y_true})', linewidth=2)
        ax.set_xlabel('Network Output ($\hat{y}$)')
        ax.set_ylabel('Loss Penalty')
        ax.fill_between(y_pred, loss, alpha=0.1, color='#4f46e5')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        
    elif selected_loss == "Mean Absolute Error (MAE)":
        st.latex(r"MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|")
        st.info("**Application:** Regression. Highly robust to outliers compared to MSE. Yields a constant gradient.")
        
        with st.sidebar:
            st.markdown("### Evaluation Parameters")
            y_true = st.number_input("Ground Truth (y)", value=0.0)
            
        y_pred = np.linspace(-10, 10, 200)
        loss = np.abs(y_true - y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_pred, loss, label='MAE Gradient Path', color='#10b981', linewidth=3)
        ax.axvline(x=y_true, color='#ef4444', linestyle='--', label=f'Target ({y_true})', linewidth=2)
        ax.set_xlabel('Network Output ($\hat{y}$)')
        ax.set_ylabel('Loss Penalty')
        ax.fill_between(y_pred, loss, alpha=0.1, color='#10b981')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        
    elif selected_loss == "Binary Cross-Entropy (BCE)":
        st.latex(r"BCE = - \frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]")
        st.info("**Application:** Binary Classification. Exploits logarithmic scales to aggressively penalize highly confident but incorrect predictions.")
        
        with st.sidebar:
            st.markdown("### Evaluation Parameters")
            y_true = st.radio("Ground Truth Class (y)", [0, 1], index=1)
            
        y_pred = np.linspace(0.001, 0.999, 200)
        if y_true == 1:
            loss = -np.log(y_pred)
        else:
            loss = -np.log(1 - y_pred)
            
        fig, ax = plt.subplots(figsize=(10, 5))
        loss_color = '#f59e0b'
        ax.plot(y_pred, loss, label=f'BCE Loss (y={y_true})', color=loss_color, linewidth=3)
        ax.set_xlabel('Predicted Probability ($\hat{y}$)')
        ax.set_ylabel('Logarithmic Penalty')
        ax.fill_between(y_pred, loss, alpha=0.1, color=loss_color)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
