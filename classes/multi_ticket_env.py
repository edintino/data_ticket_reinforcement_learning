import numpy as np

class MultiTicketEnv:
    """
    A 3-ticket offering environment.
    State: vector of size 6
        - # nbr times ticket was not bought
        - # nbr times ticket 1 was bought
        - # nbr times ticket 2 was bought
        - # nbr times ticket 3 was bought
        - remaining data
        - days since last ticket

    Action: categorical variable with 4 possibilities
        - for each ticket, you can:
        - 0 = not offer
        - 1 = offer
        - Furthermore only one of the possible tickets can be offered at a time
    """
    def __init__(self, data_tickets, data_usage, random_state=5893):
        np.random.seed(random_state)
        # Data
        self.data_usage = data_usage
        self.data_tickets = data_tickets
        
        # Instance attributes
        self.cur_step = None
        self.last_ticket = None
        self.bought_tickets = None
        self.last_reward = None
        self.avg_days = None
        
        self.state_dim = 6
        self.n_step = data_usage.shape[0]
    
        # Action permutations
        self.action_list = np.diag(np.ones(data_tickets.shape[0])).astype(int)
        
        self.action_space = np.arange(data_tickets.shape[0])
        self.reset()
        
    def offer(self, action, train=True):
        """The agent takes the action among the
        possible tickets. Furthermore calculates
        the reward associated with it."""
        
        # Update price, i.e. go to the next day
        data_change = self.data_usage[self.cur_step+1] - self.data_usage[self.cur_step]
        
        # Days counter since last ticket bought
        # Average day counter between tickets
        if data_change <= 0:
            self.last_ticket += 1
        else:
            if train:
                if sum(self.bought_tickets) == 0:
                    self.bought_tickets = self.last_ticket
                else:
                    self.avg_days = sum(self.bought_tickets)/(sum(self.bought_tickets)+1) * self.avg_days + self.last_ticket / (sum(self.bought_tickets)+1)
            self.last_ticket = 0
        
        # Reward average days number of points if ticket is well suggested
        # Reward 1 point if no ticket offers was correct
        # Else 0 reward
        dist = self.data_tickets['size']-data_change
        boolean = (dist.abs() == dist.abs().min()).astype(int).values
        
        if np.array_equal(action, boolean) and action[0] != 1:
            self.last_reward = round(self.avg_days) # weight by average days of tickets
        elif np.array_equal(action, boolean) and action[0] == 1:
            self.last_reward = 1
        else:
            self.last_reward = 0
        
        # Choose the bought ticket
        self.bought_tickets[boolean.astype(bool)] += 1
        
        # Next step
        self.cur_step += 1

        # Reward is the increase in porfolio value
        reward = self.last_reward
        
        # Done if we have run out of data
        done = self.cur_step == self.n_step - 1

        return self._get_obs(), reward, done
    

    def reset(self):
        """Resets the enviroment to its original state.
        Note that average days is now an educated guess
        at start, which is than the average after the
        first purchase. The educated guess could be based
        on the average ticket distance in days of the
        entire customer base in a real life scenario."""
        self.cur_step = 0
        self.last_ticket = 0
        self.last_reward = 0
        self.avg_days = np.random.randint(low=15,high=20)
        self.bought_tickets = np.zeros(len(self.data_tickets))

        return self._get_obs()

    def _get_obs(self):
        """Returns current state."""
        obs = np.empty(self.state_dim)
        obs[:len(self.bought_tickets)] = self.bought_tickets
        obs[-2] = self.data_usage[self.cur_step]
        obs[-1] = self.last_ticket
        return obs