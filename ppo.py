import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm


import wandb  # 新增
from torch.optim.lr_scheduler import StepLR  # 新增

import debugpy
try:
    debugpy.listen(("localhost", 9516))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

use_wandb = False
# 添加 wandb 登录检查和处理
if not wandb.api.api_key:
    try:
        wandb.login()
    except wandb.errors.AuthenticationError:
        # 替换成你的 API token
        wandb.login(key='68895610de87b04530f43ebd851357a1b12f0633')
    except Exception as e:
        print(f"Failed to log in to wandb: {e}")
        print("Will continue without wandb logging")
        use_wandb = False
else:
    use_wandb = True



class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict, episode_num):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # if use_wandb and int(episode_num * self.epochs + _) % 5 == 0:
            #     wandb.log({
            #         "actor_loss": actor_loss.item(),
            #         "critic_loss": critic_loss.item(),
            #         "return": len(transition_dict['rewards'])
            #     }, step=int(episode_num * self.epochs + _))  # 修改：使用 step 参数

    
        if use_wandb and int(episode_num) % 5 == 0:
            wandb.log({
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "return": len(transition_dict['rewards'])
            }, step=int(episode_num))  # 修改：使用 step 参数


actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 2000
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


run_name = f"ppo_v2_2000_epoch_10"  # 新增
if use_wandb:
    wandb.init(
        project="policy-gradient-cartpole",
        name=run_name,
        config={
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "gamma": gamma,
            "hidden_dim": hidden_dim,
            "num_episodes": num_episodes
        }
    )

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

# return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict,  (num_episodes / 10 * i + i_episode + 1))
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()


# 在训练完成后,添加以下代码

# 1. 保存模型
torch.save(agent.actor.state_dict(), 'ppo_actor.pth')
torch.save(agent.critic.state_dict(), 'ppo_critic.pth')

# 2. 测试环境渲染
def evaluate_policy(env, agent, episodes=5):
    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            env.render()  # 直接调用render
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        print(f"Episode Reward: {episode_reward}")
    env.close()

# 3. 执行可视化评估
evaluate_policy(env, agent)
