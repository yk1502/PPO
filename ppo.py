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
        action = torch.multinomial(act_mat, num_samples=1, replacement=True)
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
        self.act_prob_traj = []
        self.rew_traj = []
        self.new_obs_traj = []
        self.done_traj = []   
        self.values_traj = []

        self.disc_ret_traj = []
        self.adv_traj = []


    def shuffle_mem(self):
        combined = list(zip(self.obs_traj, self.act_traj, self.act_prob_traj, self.rew_traj, self.new_obs_traj, self.done_traj, self.values_traj, self.disc_ret_traj, self.adv_traj))
        random.shuffle(combined)
        obs_traj, act_traj, act_prob_traj, rew_traj, new_obs_traj, done_traj, values_traj, disc_ret_traj, adv_traj = zip(*combined)

        self.obs_traj = list(obs_traj)
        self.act_traj = list(act_traj)
        self.act_prob_traj = list(act_prob_traj)
        self.rew_traj = list(rew_traj)
        self.new_obs_traj = list(new_obs_traj)
        self.done_traj = list(done_traj)    
        self.values_traj = list(values_traj)
        self.disc_ret_traj = list(disc_ret_traj)
        self.adv_traj = list(adv_traj)

    def store_mem(self, obs, act, act_prob, rew, new_obs, done, value):
        self.obs_traj.append(obs)
        self.act_traj.append(act)
        self.act_prob_traj.append(act_prob)
        self.rew_traj.append(rew)
        self.new_obs_traj.append(new_obs)
        self.done_traj.append(done)
        self.values_traj.append(value)

    def clear_mem(self):
        self.obs_traj.clear()
        self.act_traj.clear()
        self.act_prob_traj.clear()
        self.rew_traj.clear()
        self.new_obs_traj.clear()
        self.done_traj.clear()
        self.values_traj.clear()
        self.disc_ret_traj.clear()
        self.adv_traj.clear()

    def get_mem(self):
        return torch.tensor(self.obs_traj),  \
                torch.tensor(self.act_traj),  \
                torch.tensor(self.act_prob_traj),  \
                torch.tensor(self.rew_traj),  \
                torch.tensor(self.new_obs_traj),  \
                torch.tensor(self.done_traj),  \
                torch.stack(self.values_traj),  \
                torch.tensor(self.disc_ret_traj),  \
                torch.tensor(self.adv_traj)





class PPO:
    def __init__(self):
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()
        self.memory = Memory()

        self.gamma = 0.99
        self.lamda = 0.95
        self.epsilon = 0.2

        self.steps = 50
        self.learning_rate = 3e-4

        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optim = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def get_action(self, obs):
        action, act_mat = self.policy_net.p(obs)
        return action, act_mat
    
    def calc_return(self):
        for i in range(len(self.memory.rew_traj)):
            returns = 0
            for j in range(i, len(self.memory.rew_traj)):
                if self.memory.done_traj[j] == 1:
                    break
                else:
                    returns += self.memory.rew_traj[j] * (self.gamma ** (j - i))
            self.memory.disc_ret_traj.append(returns)

    
    def calc_advantage(self):
        for i in range(len(self.memory.rew_traj)):
            advantage = 0
            for j in range(i, len(self.memory.rew_traj)):
                if self.memory.done_traj[j] == 1:
                    break
                else:
                    delta = self.memory.rew_traj[j] - self.memory.values_traj[j].item() + self.gamma * self.memory.values_traj[j + 1].item()
                    advantage += delta * ((self.gamma * self.lamda) ** (j - i))
            self.memory.adv_traj.append(advantage)

    def train(self):
        
        self.calc_return()
        self.calc_advantage()
        self.memory.shuffle_mem()

        obs, act, act_prob, rew, new_obs, done, _, disc_ret, adv = self.memory.get_mem()
        act = act.unsqueeze(1)
        disc_ret = disc_ret.unsqueeze(1)

        for step in range(self.steps):
            _, old_act_prob = self.get_action(obs.numpy())
            ratio = act_prob / old_act_prob.gather(1, act).squeeze(1)
            policy_loss = torch.min(ratio * adv, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            value = self.value_net.v(obs.numpy())
            value_loss = self.loss_fn(value, disc_ret)
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()

        self.memory.clear_mem()
        




env = gym.make('CartPole-v1')
obs, _ = env.reset()

# Setup
agent = PPO()
episodes = 1000


for ep in range(episodes):

    total_rew = 0

    while True:
        
        action, act_mat = agent.get_action(obs)
        value = agent.value_net.v(obs)

        new_obs, rew, terminated, truncated, info = env.step(action.item())
        agent.memory.store_mem(obs, action.item(), act_mat[action].item(), rew, new_obs, terminated or truncated, value)

        total_rew += rew

        obs = new_obs

        if terminated or truncated:
            observation, info = env.reset()
            print(f'episode : {ep + 1} | total reward : {total_rew}')
            break
    
    if ((ep + 1) % 5 == 0):
        agent.train()
            
