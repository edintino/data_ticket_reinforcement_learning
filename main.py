import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from classes.agent import DQNAgent
from classes.linear_model import LinearModel
from classes.multi_ticket_env import MultiTicketEnv
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.pyplot as plt

def test_plots(data,offers,data_dict,name):
    """Plots to understand better when and what
    does the agent offers after training."""

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=list(range(len(data))),
                             y=data,
                             hovertemplate='<i>Day</i>: %{x:.d}<br>'+\
                                           '<b>Remaining data</b>: %{y:.1f}',
                             name=f'Simulated {name} data'))

    fig.add_trace(go.Scatter(x=list(range(1,len(offers)+1)),
                             y=offers,
                             hovertemplate='%{text}',
                             text=[data_dict[i] for i in offers],
                             name='Offer'),
                  secondary_y=True)

    fig.layout.update(title=f'Performance on {name} data',
                      xaxis=dict(title='Days'),
                      yaxis=dict(title='Data usage in MB'),
                      legend=dict(x=.75, y=1.175))

    fig.update_yaxes(title='',
                     tickvals=list(data_dict.keys()),
                     ticktext=list(data_dict.values()),
                     secondary_y=True)

    return fig

def get_scaler(env):
    """State scaler."""

    states = []
    for _ in range(env.n_step):
        action = env.action_list[np.random.choice(env.action_space)]
        state, reward, done = env.offer(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def play_one_episode(agent, env, test_env, is_train):
    """Plays one episode. When is_train is true and we iterate
    through the train set it trains the agent and the model also.
    Meanwhile in all the other cases it goes through the data
    without any training.

    In train mode we return the cumulative reward after the episode
    for both test and train set. On the other hand if we are not in
    train mode the choices are returned to understand the choices
    the agent makes."""

    state = env.reset()
    state = scaler.transform([state])
    test_state = test_env.reset()
    test_state = scaler.transform([test_state])
    
    done = False
    test_done = False
    
    total_reward = []
    test_total_reward = []
    actions = []
    test_actions = []
    
    if is_train:
    # train mode
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.offer(env.action_list[action],False)
            next_state = scaler.transform([next_state])
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward.append(reward)
            
            test_env.avg_days = env.avg_days
            
        while not test_done:
            action = agent.act(test_state)
            test_state, reward, test_done = test_env.offer(test_env.action_list[action],True)
            test_state = scaler.transform([test_state])
            test_total_reward.append(reward)
        
        return sum(total_reward),sum(test_total_reward)
        
    elif not is_train:
    # non train mode
        while not done:
            action = agent.act(state)
            state, reward, done = env.offer(env.action_list[action],True)
            state = scaler.transform([state])
            actions.append(action)
            
        while not test_done:
            action = agent.act(test_state)
            test_state, reward, test_done = test_env.offer(test_env.action_list[action],True)
            test_state = scaler.transform([test_state])
            test_actions.append(action)
        
        return actions, test_actions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    parser.add_argument('-t', '--trainData', type=str, required=True,
                        help='name of train data .csv in the data folder')
    parser.add_argument('-v', '--testData', type=str, required=True,
                        help='name of test data .csv in the data folder')
    parser.add_argument('-eps', '--episodes', type=int, required=True,
                        help='number of episodes')
    parser.add_argument('-em', '--epsilonMin', type=float, required=True,
                        help='minimum epsilon to decay to')
    parser.add_argument('-ed', '--epsilonDecay', type=float, required=True,
                        help='decay rate of epsilon')
    parser.add_argument('-g', '--gamma', type=float, required=True,
                        help='discount rate of the reward')
    parser.add_argument('-s', '--seed', type=int, required=True,
                        help='random seed')

    args = parser.parse_args()

    data_usage = np.loadtxt(f'./data/{args.trainData}.csv',delimiter=',')
    test_data_usage = np.loadtxt(f'./data/{args.testData}.csv',delimiter=',')
    data_tickets = pd.read_csv('./data/data_tickets.csv')

    num_episodes = args.episodes

    env = MultiTicketEnv(data_tickets,data_usage,random_state=args.seed)
    test_env = MultiTicketEnv(data_tickets,test_data_usage,random_state=args.seed)

    state_size = env.state_dim
    action_size = len(env.action_list)

    agent = DQNAgent(state_size, action_size, gamma=args.gamma,
                     epsilon_min=args.epsilonMin, epsilon_decay=args.epsilonDecay)

    scaler = get_scaler(env)

    epoch_rewards = np.zeros((num_episodes,2))

    if args.mode == 'train':
    # train mode
        for e in range(num_episodes):
            t0 = datetime.now()
            train_val, test_val = play_one_episode(agent, env, test_env, True)
            epoch_rewards[e] = np.array([train_val,test_val])
            dt = datetime.now() - t0
            print(f"episode: {e + 1}/{num_episodes}, train cumulative reward: {train_val:.2f}, test cumulative reward: {test_val:.2f}, duration: {dt}")

        agent.save('linear_model')

        T = 25

        train_rewards = []
        test_rewards = []

        for i in range(T,len(epoch_rewards)):
            train_rewards.append(np.mean(epoch_rewards[i-T:i,0]))
            test_rewards.append(np.mean(epoch_rewards[i-T:i,1]))
            
        plt.plot(train_rewards,label='Train set')
        plt.plot(test_rewards,label='Test set')
        plt.title(f'Moving average of the cumulative reward over the past {T} episodes')
        plt.xlabel('Episode')
        plt.ylabel('Avg of the cumulative reward')
        plt.savefig('./images/cum_rew.png')
        plt.show()

    elif args.mode == 'test':
    # test mode
        agent.load('linear_model')

        train_offers, test_offers = play_one_episode(agent, env, test_env, False)

        data_dict = dict(zip(list(range(4)),
                     ['No offer']+[f'Data ticket {i+1}' for i in range(3)]))

        fig = test_plots(data_usage,train_offers,data_dict,'train')
        pyo.plot(fig,filename='./images/train_offers.html',auto_open=False)
        print("Train offer plot is saved under './images/train_offers.html'")

        fig = test_plots(test_data_usage,test_offers,data_dict,'test')
        pyo.plot(fig,filename='./images/test_offers.html',auto_open=False)
        print("Test offer plot is saved under './images/test_offers.html'")

    else:
        raise('Mode can be either "train" or "test"')