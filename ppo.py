import gymnasium as gym
import torch.nn as nn
import torch
import random


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(4, 64)
        self.linear2 = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.softmax(self.linear2(out))
        return out
    
    def p(self, obs):
        act_mat = self.forward(torch.from_numpy(obs))
        action = torch.multinomial(act_mat, num_samples=1, replacement=True).item()
        return action, act_mat
    

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(4, 64)
        self.linear2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        return out

    def v(self, x):
        out = self.forward(torch.from_numpy(x))
        return out


    
class Memory:
    def __init__(self):
        self.obs_traj = []
        self.act_traj = []
        self.rew_traj = []
        self.new_obs_traj = []
        self.done_traj = []   

        self.disc_ret_traj = []


    def shuffle_mem(self):
        combined = list(zip(self.obs_traj, self.act_traj, self.rew_traj, self.new_obs_traj, self.done_traj))
        random.shuffle(combined)
        obs_traj, act_traj, rew_traj, new_obs_traj, done_traj = zip(*combined)

        self.obs_traj = list(obs_traj)
        self.act_traj = list(act_traj)
        self.rew_traj = list(rew_traj)
        self.new_obs_traj = list(new_obs_traj)
        self.done_traj = list(done_traj)    

    def store_mem(self, obs, act, rew, new_obs, done):
        self.obs_traj.append(obs)
        self.act_traj.append(act)
        self.rew_traj.append(rew)
        self.new_obs_traj.append(new_obs)
        self.done_traj.append(done)

    def clear_mem(self):
        self.obs_traj.clear()
        self.act_traj.clear()
        self.rew_traj.clear()
        self.new_obs_traj.clear()
        self.done_traj.clear()

    def get_mem(self):
        return torch.tensor(self.obs_traj),  \
                torch.tensor(self.act_traj),  \
                torch.tensor(self.rew_traj),  \
                torch.tensor(self.new_obs_traj),  \
                torch.tensor(self.done_traj)  





class PPO:
    def __init__(self):
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()
        self.memory = Memory()

        self.gamma = 0.99

    def get_action(self, obs):
        action, act_mat = self.policy_net.p(obs)
        return action, act_mat
    
    def calc_return(self):
        for i in range(len(self.memory.rew_traj)):
            returns = 0
            for j in range(i, len(self.memory.rew_traj)):
                if self.memory.done_traj[j] == 1:
                    returns = 0
                else :
                    returns += self.memory.rew_traj[j] * (self.gamma ** (j - i))
            self.memory.disc_ret_traj.append(returns)

        
    

env = gym.make('CartPole-v1')
obs, _ = env.reset()

# Setup
agent = PPO()
episodes = 10


for ep in range(episodes):

    total_rew = 0

    while True:
        
        action, act_mat = agent.get_action.p(obs)

        obs, rew, terminated, truncated, info = env.step(action)
        total_rew += rew

        if terminated or truncated:
            observation, info = env.reset()
            print(f'episode : {ep + 1} | total reward : {total_rew}')
            break
            
