import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        
        np.random.seed(42)
        self.W_hx = np.random.randn(hidden_dim, input_dim) * 0.1
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_h = np.zeros((hidden_dim, 1))
        
        self.W_yh = np.random.randn(output_dim, hidden_dim) * 0.1
        self.b_y = np.zeros((output_dim, 1))

    def forward(self, x):
        # x shape: (seq_length, input_dim)
        seq_length = x.shape[0]
        self.h_states = np.zeros((seq_length + 1, self.hidden_dim, 1))
        self.outputs = np.zeros((seq_length, self.output_dim, 1))
        
        for t in range(seq_length):
            x_t = x[t].reshape(-1, 1)
            self.h_states[t+1] = tanh(np.dot(self.W_hx, x_t) + np.dot(self.W_hh, self.h_states[t]) + self.b_h)
            self.outputs[t] = np.dot(self.W_yh, self.h_states[t+1]) + self.b_y
            
        return self.outputs

    def train_step(self, x, y):
        # Very simple BPTT (Backpropagation Through Time) implementation for single sequence
        seq_length = x.shape[0]
        outputs = self.forward(x)
        
        # Loss (MSE at the last time step only for classification/sequence prediction)
        # Assuming y is the target for the last time step
        loss = np.mean((outputs[-1] - y)**2)
        
        d_y = outputs[-1] - y.reshape(-1, 1) # gradient of MSE
        
        dW_yh = np.dot(d_y, self.h_states[-1].T)
        db_y = d_y
        
        d_h = np.dot(self.W_yh.T, d_y)
        
        dW_hx = np.zeros_like(self.W_hx)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        
        for t in reversed(range(seq_length)):
            temp = d_h * tanh_derivative(self.h_states[t+1])
            x_t = x[t].reshape(-1, 1)
            
            dW_hx += np.dot(temp, x_t.T)
            dW_hh += np.dot(temp, self.h_states[t].T)
            db_h += temp
            
            d_h = np.dot(self.W_hh.T, temp)
            
        # Update weights
        self.W_yh -= self.lr * dW_yh
        self.b_y -= self.lr * db_y
        self.W_hx -= self.lr * dW_hx
        self.W_hh -= self.lr * dW_hh
        self.b_h -= self.lr * db_h
        
        return loss

    def predict(self, x):
        outputs = self.forward(x)
        return outputs[-1]


def app():
    st.markdown("## Recurrent Neural Network (RNN)")
    st.write("Train a simple RNN for Sequence-to-One prediction tasks. Perfect for time-series forecasting or sequence classification.")
    
    st.sidebar.markdown("### Architecture")
    hidden_dim = st.sidebar.slider("Hidden Dimension", 2, 32, 8)
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.05, 0.005)
    epochs = st.sidebar.slider("Epochs", 50, 2000, 500, 50)
    
    st.markdown("### 1. Sequential Data Ingestion")
    data_source = st.radio("Choose Data Source:", ["Upload Time-Series CSV", "Generate Sine Wave Data"], horizontal=True, key="rnn_data_src")
    
    X_train, y_train = None, None
    seq_length = 5
    
    if data_source == "Upload Time-Series CSV":
        uploaded_file = st.file_uploader("Upload CSV (1 Target Column, features will be windowed)", type=["csv"], key="rnn_csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                with st.expander("View Data Preview"):
                    st.dataframe(df.head(15), use_container_width=True)
                    
                cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                target_col = st.selectbox("Select Target Column to Predict", options=cols, index=0)
                seq_length = st.number_input("Sequence Window Length", min_value=2, max_value=50, value=5)
                
                series = df[target_col].values
                series = (series - np.mean(series)) / (np.std(series) + 1e-8) # Normalize
                
                # Create sliding windows
                X = []
                Y = []
                for i in range(len(series) - seq_length):
                    X.append(series[i:i+seq_length])
                    Y.append(series[i+seq_length])
                    
                X_train = np.array(X)[..., np.newaxis]
                y_train = np.array(Y)
                
                st.write(f"Generated {len(X_train)} sequences of length {seq_length}")
                st.session_state["rnn_mean"] = np.mean(df[target_col].values)
                st.session_state["rnn_std"] = np.std(df[target_col].values) + 1e-8
                
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
    else:
        st.write("Generating a standard Sine Wave dataset...")
        t = np.linspace(0, 50, 500)
        series = np.sin(t)
        
        # Sliding windows
        X = []
        Y = []
        for i in range(len(series) - seq_length):
            X.append(series[i:i+seq_length])
            Y.append(series[i+seq_length])
            
        X_train = np.array(X)[..., np.newaxis]
        y_train = np.array(Y)
        
        st.session_state["rnn_mean"] = 0
        st.session_state["rnn_std"] = 1 # Dummy for norm

    if X_train is not None and y_train is not None:
        st.markdown("### 2. Training")
        if st.button("🚀 Train Model", use_container_width=True, key="rnn_train"):
            with st.spinner("Training RNN on sequences..."):
                input_dim = 1
                model = SimpleRNN(input_dim, hidden_dim, 1, learning_rate)
                
                losses = []
                progress_bar = st.progress(0)
                
                num_samples = len(X_train)
                
                for epoch in range(epochs):
                    epoch_loss = 0
                    
                    # Random batch of 20 to speed up
                    idx = np.random.choice(num_samples, size=min(20, num_samples), replace=False)
                    for i in idx:
                        loss = model.train_step(X_train[i], y_train[i])
                        epoch_loss += loss
                        
                    epoch_loss /= len(idx)
                    
                    if epoch % max(1, epochs//100) == 0 or epoch == epochs - 1:
                        losses.append(epoch_loss)
                        progress_bar.progress((epoch + 1) / epochs)
                        
                st.session_state["rnn_model"] = model
                st.session_state["rnn_seq_len"] = seq_length
                
            st.success("Training Complete!")
            
            m1, m2 = st.columns(2)
            m1.metric("Final Loss (MSE)", f"{losses[-1]:.4f}")
            m2.metric("Sequence Length", f"{seq_length}")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(losses, color='#6366f1', linewidth=2, label='Training Loss')
            ax.set_title('RNN Training Convergence', fontweight='bold')
            ax.set_ylabel('Mean Squared Error')
            ax.set_xlabel('Epoch Steps')
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            

    st.markdown("---")
    st.markdown("### 3. Inference / Predict")
    if "rnn_model" in st.session_state:
        model = st.session_state["rnn_model"]
        sl = st.session_state["rnn_seq_len"]
        xm = st.session_state.get("rnn_mean", 0)
        xs = st.session_state.get("rnn_std", 1)
        
        st.write(f"Enter the preceding {sl} values to predict the next step in the sequence:")
        cols = st.columns(min(sl, 4))
        
        inputs = []
        for i in range(sl):
            c_idx = i % 4
            val = cols[c_idx].number_input(f"T-{sl-i}", value=0.0, key=f"rnn_in_{i}")
            inputs.append(val)
            
        if st.button("🔮 Predict Next Step", type="primary"):
            seq_input = np.array(inputs).reshape(-1, 1)
            
            # Scale input
            seq_input = (seq_input - xm) / xs
            
            pred_scaled = model.predict(seq_input)[0][0]
            
            # Unscale output
            pred_actual = (pred_scaled * xs) + xm
            
            st.markdown(f"### Predicted Next Value: <span style='color: #4f46e5;'>{pred_actual:.4f}</span>", unsafe_allow_html=True)
            
            # Visualize the prediction appended to the sequence
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(range(sl), inputs, marker='o', label="Historical Steps", color="#64748b")
            ax.plot([sl-1, sl], [inputs[-1], pred_actual], marker='s', label="Prediction", color="#ec4899", linewidth=2)
            ax.set_title("Sequence Forecast")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("Train an RNN model first to enable forecasting.")
