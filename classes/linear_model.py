import numpy as np

class LinearModel:
    """ A linear regression model """
    def __init__(self, input_dim, n_action):
        """Starting weights, bias and their momentum."""
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        self.vW = 0
        self.vb = 0

        self.losses = []

    def predict(self, X):
        # Make sure X is N x D
        # Matrix multiplication of the linear model
        assert(len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.05, momentum=0.25):
        """One step of stochastic gradient descent to
        train the model with momentum."""
        # Make sure X is N x D
        assert(len(X.shape) == 2)

        num_values = np.prod(Y.shape)

        # One step of gradient descent
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # Update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # Update weight and bias
        self.W += self.vW
        self.b += self.vb
        
        # MSE loss
        mse = np.mean((Yhat - Y)**2)
        self.losses.append(mse)
        
    def load_weights(self, name):
        """Load saved model."""
        npz = np.load(f'./models/{name}.npz')
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, name):
        """Save model."""
        np.savez(f'./models/{name}.npz', W=self.W, b=self.b)