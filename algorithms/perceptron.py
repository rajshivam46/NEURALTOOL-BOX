import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SimplePerceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.W = np.zeros(input_size + 1)
        self.lr = lr
        self.epochs = epochs

    def activation_fn(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        # Allow both 1D and 2D arrays
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Z = np.dot(X, self.W[1:]) + self.W[0]
        return self.activation_fn(Z)

    def fit(self, X, d):
        errors_list = []
        accuracy_list = []
        
        for _ in range(self.epochs):
            error_count = 0
            for i in range(d.shape[0]):
                x_i = X[i].reshape(1, -1)
                y_i = self.predict(x_i)[0]
                e = d[i] - y_i
                
                # Update weights: W_new = W_old + LR * Error * Input
                self.W[1:] += self.lr * e * X[i]
                self.W[0] += self.lr * e  # Bias update
                
                if e != 0:
                    error_count += 1
                    
            errors_list.append(error_count)
            # Accuracy
            acc = 1.0 - (error_count / d.shape[0])
            accuracy_list.append(acc)
            
            if error_count == 0:
                break
                
        return errors_list, accuracy_list

def app():
    st.markdown("## Perceptron Binary Classifier")
    st.write("Train a single-layer perceptron on your own CSV dataset or use the built-in defaults. Perfect for linearly separable binary classification tasks.")
    
    # --- Sidebar Configuration ---
    st.sidebar.markdown("### Model Hyperparameters")
    learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
    epochs = st.sidebar.slider("Max Epochs", 1, 500, 50)
    
    # --- Data Upload Section ---
    st.markdown("### 1. Data Ingestion")
    
    data_source = st.radio("Choose Data Source:", ["Upload CSV", "Use Default Logic Gates"], horizontal=True)
    
    df = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    else:
        gate_type = st.selectbox("Select Default Dataset", ["AND Gate", "OR Gate", "XOR Gate (Will fail)"])
        if gate_type == "AND Gate":
            df = pd.DataFrame({"Input1": [0,0,1,1], "Input2": [0,1,0,1], "Output": [0,0,0,1]})
        elif gate_type == "OR Gate":
            df = pd.DataFrame({"Input1": [0,0,1,1], "Input2": [0,1,0,1], "Output": [0,1,1,1]})
        else:
            df = pd.DataFrame({"Input1": [0,0,1,1], "Input2": [0,1,0,1], "Output": [0,1,1,0]})
            
    if df is not None:
        with st.expander("View Dataset Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
        # Select Features and Target
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            st.error("No numeric columns found in the dataset.")
            return
            
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select Target Column (Y)", options=cols, index=len(cols)-1)
        with col2:
            feature_cols = st.multiselect("Select Feature Columns (X)", options=[c for c in cols if c != target_col], default=[c for c in cols if c != target_col])
        
        if not feature_cols:
            st.warning("Please select at least one feature column.")
            return

        X = df[feature_cols].values
        y = df[target_col].values
        
        # Ensure binary target
        unique_targets = np.unique(y)
        if len(unique_targets) > 2:
            st.error(f"Perceptron requires binary targets. Found {len(unique_targets)} unique values: {unique_targets}")
            return
            
        # Map targets to 0 and 1 if they are not already
        if not set(unique_targets).issubset({0, 1}):
            y = np.where(y == unique_targets[0], 0, 1)
            st.info(f"Target values mapped for binary classification: {unique_targets[0]} -> 0, {unique_targets[1]} -> 1")

        st.markdown("### 2. Training")
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Training Perceptron..."):
                model = SimplePerceptron(input_size=X.shape[1], lr=learning_rate, epochs=epochs)
                errors, accuracies = model.fit(X, y)
                
                # Save model to session state
                st.session_state["perceptron_model"] = model
                st.session_state["perceptron_features"] = feature_cols
                st.session_state["perceptron_targets"] = unique_targets
                
            st.success(f"Training completed in {len(errors)} epochs!")
            
            # --- Metrics ---
            m1, m2, m3 = st.columns(3)
            final_acc = accuracies[-1] * 100
            m1.metric(label="Final Accuracy", value=f"{final_acc:.1f}%")
            m2.metric(label="Final Misclassifications", value=f"{errors[-1]}")
            m3.metric(label="Total Epochs Run", value=f"{len(errors)}")

            # --- Visualizations ---
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            
            # Error plot
            ax[0].plot(range(1, len(errors)+1), errors, marker='o', color='#4f46e5', linewidth=2)
            ax[0].set_title('Training Error Over Time', fontweight='bold')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Misclassifications')
            ax[0].grid(True, linestyle='--', alpha=0.7)
            
            # Accuracy plot
            ax[1].plot(range(1, len(accuracies)+1), [a*100 for a in accuracies], marker='s', color='#10b981', linewidth=2)
            ax[1].set_title('Training Accuracy Over Time', fontweight='bold')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy (%)')
            ax[1].set_ylim([-5, 105])
            ax[1].grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)

    st.markdown("---")
    # --- Prediction Section ---
    st.markdown("### 3. Inference / Prediction")
    if "perceptron_model" in st.session_state:
        st.write("Provide values for the features to get a prediction utilizing the trained model.")
        
        feature_cols_saved = st.session_state["perceptron_features"]
        model_saved = st.session_state["perceptron_model"]
        targets_saved = st.session_state["perceptron_targets"]
        
        # Dynamic input fields
        cols = st.columns(min(len(feature_cols_saved), 4))
        input_data = []
        for i, f in enumerate(feature_cols_saved):
            col_idx = i % 4
            val = cols[col_idx].number_input(f"Value for {f}", value=0.0)
            input_data.append(val)
            
        if st.button("🔮 Predict", type="primary"):
            input_array = np.array(input_data)
            prediction_bin = model_saved.predict(input_array)[0]
            
            # Map back to original target if needed
            if len(targets_saved) == 2 and not set(targets_saved).issubset({0, 1}):
                predicted_class = targets_saved[1] if prediction_bin == 1 else targets_saved[0]
            else:
                predicted_class = prediction_bin
                
            st.markdown(f"### Predicted Class: <span style='color: #4f46e5;'>{predicted_class}</span>", unsafe_allow_html=True)
    else:
        st.info("Train a model first to enable the Inference section.")