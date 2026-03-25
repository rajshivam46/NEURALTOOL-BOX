import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.lr = lr
        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.uniform(-1, 1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.uniform(-1, 1, size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
        
    def backward(self, X, y):
        m = X.shape[0]
        # Output layer error
        error = y - self.a2
        d_output = error * sigmoid_derivative(self.a2)
        
        # Hidden layer error
        error_hidden = np.dot(d_output, self.W2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.a1)
        
        # Updates
        self.W2 += np.dot(self.a1.T, d_output) * self.lr / m
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr / m
        self.W1 += np.dot(X.T, d_hidden) * self.lr / m
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr / m
        
        return np.mean(np.square(error))

    def predict(self, X):
        preds = self.forward(X)
        return np.where(preds >= 0.5, 1, 0)

def app():
    st.markdown("## Multi-Layer Perceptron (Backpropagation)")
    st.write("Train a neural network with one hidden layer using gradient descent Backpropagation.")
    
    # --- Sidebar Configuration ---
    st.sidebar.markdown("### Network Architectures & Hyperparameters")
    hidden_nodes = st.sidebar.slider("Hidden Layer Nodes", 2, 64, 8)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 2.0, 0.5, 0.01)
    epochs = st.sidebar.slider("Epochs", 100, 10000, 1000, 100)
    
    # --- Data Ingestion ---
    st.markdown("### 1. Data Ingestion")
    data_source = st.radio("Choose Data Source:", ["Upload CSV", "Use Default Non-linear (XOR)"], horizontal=True, key="bp_data_src")
    
    df = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV dataset (Binary Classification recommended)", type=["csv"], key="bp_csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    else:
        df = pd.DataFrame({"Input1": [0,0,1,1], "Input2": [0,1,0,1], "Output (XOR)": [0,1,1,0]})
            
    if df is not None:
        with st.expander("View Dataset Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            st.error("No numeric columns found.")
            return
            
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select Target Column (Y)", options=cols, index=len(cols)-1, key="bp_tc")
        with col2:
            feature_cols = st.multiselect("Select Feature Columns (X)", options=[c for c in cols if c != target_col], default=[c for c in cols if c != target_col], key="bp_fc")
        
        if not feature_cols:
            st.warning("Please select at least one feature.")
            return

        X_raw = df[feature_cols].values
        y_raw = df[target_col].values
        
        # Ensure binary target
        unique_targets = np.unique(y_raw)
        if len(unique_targets) != 2:
            st.warning(f"Warning: MLP expects binary targets [0, 1]. Found {len(unique_targets)} unique values. Will attempt mapping.")
            if len(unique_targets) == 1:
                st.error("Need at least 2 classes.")
                return

        # Map targets to 0, 1
        y = np.where(y_raw == unique_targets[0], 0, 1).reshape(-1, 1)
        
        # Standardize features to help convergence
        X_mean = np.mean(X_raw, axis=0)
        X_std = np.std(X_raw, axis=0) + 1e-8
        X = (X_raw - X_mean) / X_std

        st.markdown("### 2. Training")
        if st.button("🚀 Train Model", use_container_width=True, key="bp_train"):
            with st.spinner("Training Neural Network via Backpropagation..."):
                model = MLP(input_dim=X.shape[1], hidden_dim=hidden_nodes, output_dim=1, lr=learning_rate)
                
                losses = []
                accuracies = []
                progress_bar = st.progress(0)
                
                for i in range(epochs):
                    # Forward pass is required before backward pass
                    model.forward(X)
                    loss = model.backward(X, y)
                    
                    if i % max(1, epochs//100) == 0 or i == epochs - 1:
                        preds = model.predict(X)
                        acc = np.mean(preds == y)
                        losses.append(loss)
                        accuracies.append(acc)
                        progress_bar.progress((i + 1) / epochs)
                
                # Save models and preprocessors to session
                st.session_state["bp_model"] = model
                st.session_state["bp_mean"] = X_mean
                st.session_state["bp_std"] = X_std
                st.session_state["bp_features"] = feature_cols
                st.session_state["bp_targets"] = unique_targets

            st.success("Training Complete!")
            
            # --- Metrics ---
            m1, m2 = st.columns(2)
            m1.metric("Final Accuracy", f"{accuracies[-1]*100:.1f}%")
            m2.metric("Final Loss (MSE)", f"{losses[-1]:.4f}")
            
            # --- Chart ---
            fig, ax1 = plt.subplots(figsize=(10, 4))
            color = '#ef4444'
            ax1.set_xlabel('Epoch Steps')
            ax1.set_ylabel('Loss (MSE)', color=color)
            ax1.plot(losses, color=color, linewidth=2, label='Loss')
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = '#10b981'
            ax2.set_ylabel('Accuracy', color=color)
            ax2.plot(accuracies, color=color, linewidth=2, label='Accuracy', linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)
            
            fig.tight_layout()
            st.pyplot(fig)

    st.markdown("---")
    # --- Prediction ---
    st.markdown("### 3. Inference / Predict")
    if "bp_model" in st.session_state:
        st.write("Predict new values using the trained neural network.")
        model = st.session_state["bp_model"]
        f_cols = st.session_state["bp_features"]
        t_vals = st.session_state["bp_targets"]
        x_m = st.session_state["bp_mean"]
        x_s = st.session_state["bp_std"]
        
        cols = st.columns(min(len(f_cols), 4))
        input_data = []
        for i, f in enumerate(f_cols):
            c_idx = i % 4
            val = cols[c_idx].number_input(f"Value for {f}", value=0.0, key=f"bp_in_{i}")
            input_data.append(val)
            
        if st.button("🔮 Predict", type="primary", key="bp_predict"):
            raw_input = np.array(input_data).reshape(1, -1)
            # Scale input
            scaled_input = (raw_input - x_m) / x_s
            pred_prob = model.forward(scaled_input)[0][0]
            pred_class_idx = 1 if pred_prob >= 0.5 else 0
            
            predicted_label = t_vals[pred_class_idx]
            
            st.markdown(f"### Predicted Class: <span style='color: #4f46e5;'>{predicted_label}</span>", unsafe_allow_html=True)
            st.write(f"Activation Probability (Confidence): {pred_prob:.4f}")
    else:
        st.info("Train a model first to enable predictions.")
