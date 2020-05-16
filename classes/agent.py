import numpy as np
from .linear_model import LinearModel

class DQNAgent(object):
    """Agent with decaying discovery rate, epsilon.
    The reward discount is given by gamma."""
    def __init__(self, state_size, action_size,
                 gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon    # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])    # returns action


    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target
        
        # Run one training step
        self.model.sgd(state, target_full)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)
        self.epsilon = self.epsilon_min

    def save(self, name):
        self.model.save_weights(name)