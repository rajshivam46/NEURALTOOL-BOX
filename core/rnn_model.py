import numpy as np

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
        seq_length = x.shape[0]
        self.h_states = np.zeros((seq_length + 1, self.hidden_dim, 1))
        self.outputs = np.zeros((seq_length, self.output_dim, 1))
        
        for t in range(seq_length):
            x_t = x[t].reshape(-1, 1)
            self.h_states[t+1] = tanh(np.dot(self.W_hx, x_t) + np.dot(self.W_hh, self.h_states[t]) + self.b_h)
            self.outputs[t] = np.dot(self.W_yh, self.h_states[t+1]) + self.b_y
            
        return self.outputs

    def train_step(self, x, y):
        seq_length = x.shape[0]
        outputs = self.forward(x)
        
        loss = np.mean((outputs[-1] - y)**2)
        d_y = outputs[-1] - y.reshape(-1, 1)
        
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
            
        self.W_yh -= self.lr * dW_yh
        self.b_y -= self.lr * db_y
        self.W_hx -= self.lr * dW_hx
        self.W_hh -= self.lr * dW_hh
        self.b_h -= self.lr * db_h
        
        return loss

    def predict(self, x):
        outputs = self.forward(x)
        return outputs[-1]
