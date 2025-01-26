- PolicyNet
  - 两层网络
  - 输出之前进行 softmax
  - state_dim: 4
  - action_dim: 2
  - hidden_dim: 128

- REINFORCE
  - 关键构成
    - policy_net: PolicyNet
    - optimizer: 用来优化 policy_net
    - gamma: 折扣因子
  - 函数:
    - take_action: 输入 state, 由 poliy 输出分布，并根据概率分布随机采样, 输出 action
    - update: 
      
       
       


