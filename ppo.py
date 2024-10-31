import gymnasium as gym
import torch.nn as nn
import torch
import random


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(4, 32)
        self.linear2 = nn.Linear(32, 2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.softmax(self.linear2(out))
        return out
    
    def p(self, obs, actions=None):
        act_prob = self.forward(obs)
        dist = torch.distributions.Categorical(act_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        if (actions != None):
            return actions, dist.log_prob(actions)

        return action, log_prob
    

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(4, 32)
        self.linear2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        return out

    def v(self, x):
        out = self.forward(x)
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
        return torch.stack(self.obs_traj),  \
                torch.tensor(self.act_traj), \
                torch.tensor(self.act_prob_traj),  \
                torch.tensor(self.disc_ret_traj), \
                torch.tensor(self.adv_traj)





class PPO:
    def __init__(self):
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()
        self.memory = Memory()

        self.gamma = 0.99
        self.lamda = 0.95
        self.epsilon = 0.1

        self.steps = 30
        self.learning_rate = 0.01

        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optim = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def pred(self, obs, actions=None):
        action, log_prob = self.policy_net.p(obs, actions)
        value = self.value_net.v(obs)
        return action, log_prob, value
    
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
                    delta = self.memory.rew_traj[j] - self.memory.values_traj[j] + self.gamma * self.memory.values_traj[j + 1]
                    advantage += delta * ((self.gamma * self.lamda) ** (j - i))
            self.memory.adv_traj.append(advantage)


    def train(self):
        
        self.calc_return()
        self.calc_advantage()
        self.memory.shuffle_mem()

        # action, old_log_prob, disc_ret, adv are in 1d shape
        # obs in 2d shape (traj_size, obs_size)
        obs, actions, old_log_prob, disc_ret, adv = self.memory.get_mem()

        for step in range(self.steps):
            _, log_prob, value = self.pred(obs, actions)
            ratio = (log_prob - old_log_prob).exp()
            policy_loss_1 = ratio * adv
            policy_loss_2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            value_loss = self.loss_fn(value.flatten(), disc_ret)
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()



        self.memory.clear_mem()
        



env = gym.make('CartPole-v1')
obs, _ = env.reset() # obs in numpy
episodes = 10000
agent = PPO()


for ep in range(episodes):
    total_rew = 0

    while True:
        action, log_prob, value = agent.pred(torch.from_numpy(obs))
        new_obs, rew, terminated, truncated, info = env.step(action.item())

        # have to store either in pytorch tensor, or python data type format, no numpy
        # the obs and new_obs is in tensor, example : torch.tensor([1, 2 ,3]), stored in a python array
        # the tensors are then stacked up together
        agent.memory.store_mem(torch.from_numpy(obs), action.item(), log_prob, rew, torch.from_numpy(new_obs), terminated or truncated, value.item())
        obs = new_obs
        total_rew += rew

        if (terminated or truncated):
            print(f"episode : {ep + 1} | total reward : {total_rew}")
            obs, _ = env.reset() # obs in numpy
            break

    if (ep+1) % 5 == 0:
        agent.train()

