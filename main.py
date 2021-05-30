from  ddpg_agent import Agent
from config import Config
from collections import deque
from cli import Cli
import numpy as np
import torch
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
import pandas as pd
import os
import sys

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

cli = Cli()
args = cli.parse()

env = UnityEnvironment(file_name="./Unity/Reacher.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]


def train(episodes=1000,max_t=1000):
    try:
        agent = Agent(state_size=33, action_size=4, random_seed=0)
        scores = []                      
        scores_window = deque(maxlen=100)  
        config = Config()
        eps = config.EPS_START

        for i_episode in range(1, episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations
            agent.reset()
            score = np.zeros(1) 
            for _ in range(max_t +1):
                action = agent.act(state,eps)
                env_info = env.step(action)[brain_name]       
                next_state = env_info.vector_observations   # get the next state
                reward = env_info.rewards                   # get the reward
                done = env_info.local_done                  # see if episode has finished

                score += env_info.rewards

                agent.step(state, action, reward, next_state, done)
                state = next_state
                eps = eps - config.LIN_EPS_DECAY
                eps = np.maximum(eps, config.EPS_END)

                if np.any(done):
                    break 

            scores_window.append(score)      
            scores.append(score)              

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 2 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            mean = np.mean(scores_window)
            if mean > 30.0 and mean <= 31.5:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(agent.actor_local.state_dict(), 'trained_model.pth')
                torch.save(agent.critic_local.state_dict(), 'trained_model.pth')

        torch.save(agent.actor_local.state_dict(), 'trained_model.pth')
        torch.save(agent.critic_local.state_dict(), 'trained_model.pth')
        return scores

    except KeyboardInterrupt:
        torch.save(agent.actor_local.state_dict(), 'trained_model.pth')
        torch.save(agent.critic_local.state_dict(), 'trained_model.pth')
        plot_score_chart(scores)
        sys.exit(0)

def load_model(model,path='trained_model.pth'):
    model.load_state_dict(torch.load(os.path.join(THIS_FOLDER, path)))

def test():
        agent = Agent(state_size=33, action_size=4, seed=0)
        load_model(agent.qnetwork_local)    
        env_info = env.reset(train_mode=False)[brain_name]

        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state, 0)
            env_info = env.step(int(action))[brain_name]       
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            state = next_state
            score += reward
            if done:
                print('\r\tTest Score: {:.2f}'.format( score, end=""))
                break 

def print_env_info(env):
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

def plot_score_chart(scores):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.title("Scores")
    rolling_mean = pd.Series(scores).rolling(100).mean()
    plt.plot(rolling_mean)
    plt.show()

print_env_info(env)

if args.train:
    scores = train()
    plot_score_chart(scores)
else:
    scores = test()

    
