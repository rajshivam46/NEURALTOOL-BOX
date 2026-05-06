import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid
import sys
import os

# Add the root directory to sys.path so we can import from backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.models.perceptron.model import SimplePerceptron
from backend.models.backprop.model import MLP
from backend.models.rnn.model import SimpleRNN
from backend.utils.data_utils import validate_binary_target, standardize_data
from backend.utils.stats_store import record_model_run, increment_datasets, get_dashboard_stats

st.set_page_config(page_title="Neural Toolbox", layout="wide")

# Initialize session state for data
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Data Upload", "Perceptron", "Backpropagation", "RNN", "CNN (Face Recognition)"])

if page == "Dashboard":
    st.title("Neural Toolbox Dashboard")
    stats = get_dashboard_stats()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Datasets Uploaded", stats["total_datasets"])
    col2.metric("Models Trained", stats["models_trained"])
    
    st.subheader("Recent Runs")
    if stats["recent_runs"]:
        st.table(stats["recent_runs"])
    else:
        st.info("No models trained yet.")

elif page == "Data Upload":
    st.title("Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            increment_datasets()
            st.success("File uploaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            
    if st.session_state.df is not None:
        st.subheader("Current Dataset Preview")
        st.dataframe(st.session_state.df.head())

elif page == "Perceptron":
    st.title("Simple Perceptron")
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        target_col = st.selectbox("Target Column", numeric_cols)
        feature_cols = st.multiselect("Feature Columns", [c for c in numeric_cols if c != target_col])
        lr = st.number_input("Learning Rate", value=0.1, step=0.01)
        epochs = st.number_input("Epochs", value=50, step=10)
        
        if st.button("Train Perceptron"):
            if not feature_cols:
                st.error("Select at least one feature column.")
            else:
                try:
                    X = df[feature_cols].values
                    y_raw = df[target_col].values
                    is_binary, unique_targets, y = validate_binary_target(y_raw)
                    
                    if not is_binary:
                        st.error("Perceptron only supports Binary targets (2 classes).")
                    else:
                        model = SimplePerceptron(input_size=X.shape[1], lr=lr, epochs=epochs)
                        with st.spinner("Training..."):
                            errors, accuracies = model.fit(X, y)
                            record_model_run("Perceptron", epochs, float(accuracies[-1]) if accuracies else 0.0, float(errors[-1]) if errors else 0.0)
                            
                        st.success("Training Complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Error over Epochs")
                            st.line_chart(errors)
                        with col2:
                            st.subheader("Accuracy over Epochs")
                            st.line_chart(accuracies)
                except Exception as e:
                    st.error(f"Error during training: {e}")

elif page == "Backpropagation":
    st.title("Backpropagation (MLP)")
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        target_col = st.selectbox("Target Column", numeric_cols)
        feature_cols = st.multiselect("Feature Columns", [c for c in numeric_cols if c != target_col])
        hidden_nodes = st.number_input("Hidden Nodes", value=8, step=1)
        lr = st.number_input("Learning Rate", value=0.1, step=0.01)
        epochs = st.number_input("Epochs", value=1000, step=100)
        
        if st.button("Train MLP"):
            if not feature_cols:
                st.error("Select at least one feature column.")
            else:
                try:
                    X_raw = df[feature_cols].values
                    y_raw = df[target_col].values
                    
                    is_binary, unique_targets, y = validate_binary_target(y_raw)
                    y = y.reshape(-1, 1)
                    X, X_mean, X_std = standardize_data(X_raw)
                    
                    model = MLP(input_dim=X.shape[1], hidden_dim=hidden_nodes, output_dim=1, lr=lr)
                    losses = []
                    accs = []
                    
                    with st.spinner("Training..."):
                        progress_bar = st.progress(0)
                        for i in range(epochs):
                            model.forward(X)
                            loss = model.backward(X, y)
                            if i % max(1, epochs // 100) == 0 or i == epochs - 1:
                                preds = model.predict(X)
                                acc = np.mean(preds == y)
                                losses.append(loss)
                                accs.append(acc)
                                progress_bar.progress((i + 1) / epochs)
                                
                    record_model_run("Backprop MLP", epochs, float(accs[-1]) if accs else 0.0, float(losses[-1]) if losses else 0.0)
                    st.success("Training Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Loss over Epochs")
                        st.line_chart(losses)
                    with col2:
                        st.subheader("Accuracy over Epochs")
                        st.line_chart(accs)
                except Exception as e:
                    st.error(f"Error during training: {e}")

elif page == "RNN":
    st.title("Recurrent Neural Network (RNN)")
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        target_col = st.selectbox("Target Column (Time Series)", numeric_cols)
        seq_length = st.number_input("Sequence Length", value=5, step=1)
        hidden_dim = st.number_input("Hidden Dimension", value=8, step=1)
        epochs = st.number_input("Epochs", value=100, step=10)
        
        if st.button("Train RNN"):
            try:
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
                
                model = SimpleRNN(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
                losses = []
                
                with st.spinner("Training..."):
                    progress_bar = st.progress(0)
                    for epoch in range(epochs):
                        epoch_loss = 0
                        idx = np.random.choice(len(X_train), size=min(10, len(X_train)), replace=False)
                        for i in idx:
                            loss = model.train_step(X_train[i], y_train[i])
                            epoch_loss += loss
                        losses.append(epoch_loss / len(idx))
                        progress_bar.progress((epoch + 1) / epochs)
                        
                record_model_run("RNN", epochs, 0.0, float(losses[-1]) if losses else 0.0)
                st.success("Training Complete!")
                st.subheader("Loss over Epochs")
                st.line_chart(losses)
            except Exception as e:
                st.error(f"Error during training: {e}")

elif page == "CNN (Face Recognition)":
    st.title("CNN Face Recognition")
    st.info("Face registration and recognition via webcam can be done here.")
    st.write("Streamlit provides a simple `st.camera_input` component.")
    
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        st.image(img_file_buffer)
        # Note: In a real implementation, you would convert the image to base64
        # and pass it to backend.models.cnn.face_recognizer functions.
        st.info("CNN backend integration would process this image.")
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello World"}
