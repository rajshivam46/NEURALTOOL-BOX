import numpy as np

class SimplePerceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.W = np.zeros(input_size + 1)
        self.lr = lr
        self.epochs = epochs

    def activation_fn(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
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
                
                self.W[1:] += self.lr * e * X[i]
                self.W[0] += self.lr * e
                
                if e != 0:
                    error_count += 1
                    
            errors_list.append(error_count)
            acc = 1.0 - (error_count / d.shape[0])
            accuracy_list.append(acc)
            
            if error_count == 0:
                break
                
        return errors_list, accuracy_list
