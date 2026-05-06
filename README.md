# 🧠 Neural Network Toolbox

Welcome to the **Neural Network Toolbox**, an educational machine learning platform designed to help you understand and interact with core neural network algorithms!

This interactive web application, built using [Streamlit](https://streamlit.io/), provides a hands-on experience with fundamental machine learning models. You can easily upload your datasets, train models from scratch, and make real-time predictions right from your browser. 

## ✨ Features

- **Interactive UI:** A highly intuitive, easy-to-use interface powered by Streamlit.
- **Custom Dataset Support:** Upload your own CSV files. The application parses and utilizes your exact original features for training and prediction.
- **Core Algorithms Implemented:**
  - **Perceptron:** Understand the foundation of neural networks with single-layer binary classifiers.
  - **Backpropagation:** Train Multi-Layer Perceptrons (MLPs) and visualize the learning process.
  - **Recurrent Neural Networks (RNN):** Explore sequence learning and standard RNN architectures.
  - **Loss Functions:** Interactively study various loss functions and their impact on model optimization.
- **End-to-End Pipeline:** Seamlessly go from Data Upload -> Model Training -> Real-time Prediction.

## 🛠️ Installation & Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd "neural tool"
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Dependencies include: `streamlit`, `numpy`, `pandas`, `matplotlib`.*

## 🚀 Usage

To start the Neural Network Toolbox, simply run the following command in your terminal:

```bash
streamlit run app.py
```

This will automatically open the application in your default web browser (usually at `http://localhost:8501`).

### How to Use the Modules:
1. **Select an Algorithm:** Choose from Perceptron, Backpropagation, RNN, or Loss Functions using the sidebar.
2. **Upload your CSV:** Provide your training data.
3. **Train:** Adjust hyperparameters (if available) and click the **Train** button to fit the network parameters.
4. **Predict:** Enter specific values for your features to get localized predictions directly from your newly trained network.

## 📁 Project Structure

```text
neural tool/
├── app.py                     # Main Streamlit application entry point
├── requirements.txt           # Python dependencies
├── pages/                     # Streamlit pages for individual modules
│   ├── 1_Perceptron.py
│   ├── 2_Backpropagation.py
│   ├── 3_RNN.py
│   └── 4_Loss_Functions.py
├── core/                      # Core neural network algorithm implementations
├── algorithms/                # Additional algorithm helpers
└── utils.py                   # Shared utility functions
```

## 🤝 Contribution

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
*Happy Learning!* 🧠🚀
