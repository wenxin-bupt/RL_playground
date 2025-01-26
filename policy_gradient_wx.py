import torch
import torch.nn.functional as F
import gym
import random
import wandb  # 新增
from torch.optim.lr_scheduler import StepLR  # 新增
import numpy as np

import debugpy
try:
    debugpy.listen(("localhost", 9515))
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

# define a network
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
# define policy gradient process
class Policy_Gradient:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate,
                 gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.device = device
        self.total_loss = 0  # 新增，用于累积一个 episode 的 loss
        self.scheduler = StepLR(self.optimizer, 
                              step_size=500,  # 每500个episode调整一次
                              gamma=0.1)      # 每次调整为原来的0.1倍

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        rand_val = random.random()
        if rand_val < probs[0][0]:
            return 0
        else:
            return 1
        

    def take_batch_action(self, state):
        # state已经是batch形式的tensor
        probs = self.policy_net(state)
        # 使用Categorical分布进行采样
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action
 
        
    def update(self, transition_dict):
        # update based on an episode
        rewards = transition_dict['rewards']

        G = 0.
        self.optimizer.zero_grad()
        total_loss = 0  # 新增，记录这个 episode 的总 loss
        # reverse rewards
        for i in range(len(rewards)):
            rev_i = len(rewards) - i - 1
            G = rewards[rev_i]  + self.gamma * G
            state = torch.tensor([transition_dict['states'][rev_i]], dtype=torch.float).to(self.device)
            action = torch.tensor([transition_dict['actions'][rev_i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            loss = -log_prob * G
            total_loss += loss.item()  # 新增
            loss.backward()
        self.optimizer.step()

        total_loss /= len(rewards)  # 新增，计算平均 loss
        return total_loss  # 新增，返回总 loss
    
    def update_batch(self, transition_dict):
        # update based on an episode

        self.optimizer.zero_grad()
        total_loss = 0  # 新增，记录这个 episode 的总 loss
        # reverse rewards
        state = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        log_prob = torch.log(self.policy_net(state).gather(1, action))
        Gs = torch.tensor(transition_dict['Gs']).view(-1, 1).to(self.device)
        loss = -log_prob * Gs
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        return loss

# setup gym environment
# rl_env = gym.make('CartPole-v0')

state_dim = 4
hidden_dim = 128
action_dim = 2
learning_rate = 1e-3
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
gamma = 0.98
pg = Policy_Gradient(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)
num_episodes = 40000
batch_size = 128
rl_env_batch = [gym.make('CartPole-v0') for _ in range(batch_size)]

# 初始化 wandb，添加实验命名
run_name = f"PG_v1"  # 新增
if use_wandb:
    wandb.init(
        project="policy-gradient-cartpole",
        group="PG_experiment",  # 新增：实验组
        config={
            "learning_rate": learning_rate,
            "gamma": gamma,
            "hidden_dim": hidden_dim,
            "num_episodes": num_episodes
        }
    )

# episode num
episode_num_loop = num_episodes // batch_size

# training process
for ii in range(episode_num_loop): 
    transition_dict = {
        'states': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }

    rl_cnt = 0
    state_list = []
    while True:
        if rl_cnt == 0:
            for bi in  range(batch_size):
                state = rl_env_batch[bi].reset()
                state_list.append(state)

        state = torch.tensor(state_list, dtype=torch.float).to(device)
        action = pg.take_batch_action(state)

        next_state_list = []
        done_list = []
        reward_list = []
        reward_cnt = []
 
        all_done = True
        for bi in  range(batch_size):
            rl_env = rl_env_batch[bi]
            next_state, reward, done, _ = rl_env.step(action[bi].item())    
            if done == 0:
                all_done = False
            next_state_list.append(next_state)
            done_list.append(done)
            reward_list.append(reward)

            

        transition_dict['states'].append(state_list)
        transition_dict['actions'].append(action)
        transition_dict['rewards'].append(reward_list)
        transition_dict['dones'].append(done_list)
        
        state_list = next_state_list
        rl_cnt += 1

        if all_done:
            break


    # do update
    # compute G
    transition_dict["Gs"] = []

    new_trasition_list = []

    Gs = [0] * batch_size
    max_reward_len = len(transition_dict["rewards"])

    dones_cnt = transition_dict["dones"]
    dones_cnt = np.array(dones_cnt).transpose()
    dones_cnt = np.sum(dones_cnt == 0, axis=1)
    dones_mean = np.mean(dones_cnt)


    for ai  in range(max_reward_len):
        rev_ai = max_reward_len - ai - 1
        rewards = transition_dict["rewards"][rev_ai]
        cur_Gs = []
        for bi in range(batch_size):
            # get rewards
            if transition_dict['dones'][rev_ai][bi] == True:
                Gs[bi] = 0.
            Gs[bi] = rewards[bi] + pg.gamma * Gs[bi]
            cur_Gs.append(Gs[bi])
        transition_dict["Gs"].append(cur_Gs)

    # update
    total_loss = 0
    total_reward = 0
    total_Gs = 0
    
    # 生成随机顺序的索引
    random_indices = np.random.permutation(max_reward_len)
    
    # 按随机顺序进行更新
    for ai in random_indices:
        new_trasition = {
            'states': transition_dict["states"][ai],
            'actions': transition_dict["actions"][ai],
            "Gs": transition_dict["Gs"][ai]
        }

        loss = pg.update_batch(new_trasition)
        total_loss += loss.item()

    # count dones
    
    

    print(f"Episode {ii}: Loss {total_loss}", "dones_mean", dones_mean)
        

            
            

        
        

        
    







    # while not done:
    #     action = pg.take_action(state)
    #     next_state, reward, done, _ = rl_env.step(action)
    #     transition_dict['states'].append(state)
    #     transition_dict['actions'].append(action)
    #     transition_dict['rewards'].append(reward)
    #     state = next_state
    
    # episode_reward = sum(transition_dict['rewards'])
    # episode_loss = pg.update(transition_dict)
    
    # # 更新学习率
    # pg.scheduler.step()

    # if ii % 50 == 0: 
    #     # 记录日志
    #     if use_wandb:
    #         # 获取当前学习率
    #         current_lr = pg.optimizer.param_groups[0]['lr']
    #         wandb.log({
    #             "episode": ii,
    #             "reward": episode_reward,
    #             "loss": episode_loss,
    #             "learning_rate": current_lr  # 新增：记录学习率
    #         })
        
    # print("Episode %d: Reward %d, LR %.6f" % (ii, episode_reward, current_lr))  # 修改print内容

# 结束 wandb
if use_wandb:
    wandb.finish()

