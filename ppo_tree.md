

- PolicyNet 
  - 两层网络
  - 输入 state 维度为 4:
    - 小车位置 x
    - 小车速度 x_
    - 杆子角度 theta
    - 杆子角速度 theta_
  - 输出 action
    - 如果是离散动空间: 
      0: 向左， 1 向右
    - 如果是连续动作空间: 
      连续的力值, [-1, 1] 的范围内，表示推力的大小和方向

- ValueNet
  - 两层网络
  - 输入 state 维度为 4:
    如上不赘述
  - 输出 value 维度为 1:


- PPO 算法 
  - 组成成分:
    - 一个 actor (PolicyNet 实例)
    - 一个 critic (ValueNet 实例)
    - gamma: ==0.98, 折扣因子, 取值 [0,1]
    - lmbda: ==0.95 用于计算优势函数, 控制即使 TD 误差和长期优势估计的平衡
    - epochs: ==10 一共训练 10 个 epochs
    - eps: ==0.2 限制策略更新幅度。防止新旧策略相差较大 (PPO 中阶段范围的参数)

- 训练逻辑
  - 500 episodes, 被拆分成 10 个小组
  - 每个小组 50 个 episodes 
  - 每个 episode 都是以 env.reset()  作为开始。 这样就会对场景进行随机重置。
  - 每个 state 通过 agent.take_action(state) 获取 action
     - 需要注意的是: 这里的 action 是基于模型输出的分布进行的随机采样
     - 每个 action 的 reward 都是 1. 直到 episode 终结。(终结条件是杆的角度，距离，最大步数满足其中的一个)。 所以比较的是不同 episode 之间的  return
     - 整个 episode 的 states, actions, next_states, rewards, done 都会被记录下来, 送给 agent.update()。 ppo 的算法就在 update 里面
    
- update 函数 (ppo 角度解释)
  - 计算 td_target  
    - td_target: 当前的 value == 每个动作的 reward + gamma * critic(next_value)
    - 每个 action 都对应一个 state, action, value
  - 计算 td_delta (也就是 advantage)
    - td_delta: value 评估之差 == td_target - critic(current_state)
  
  - 计算 ratio



一些问题:
- 设置不同的种子收敛速度会不一样么?
- 经过训练之后 value 会长什么样子?
- 开环情况下, 每一个点的 reward 应该如何计算?
- 如何利用同一条数据进行多次训练?
- 500 个 episodes 之间 LR 是如何管理的呢？ 会上升或者下降么?
- next_states 在 dppo 里面是当前 state + action 构成的
- 为什么是梯度上升?
- 策略梯度, Actor-Critic, TRPO, PPO 看起来像是这一个改进流程
- GMM 的方法呢? 通过其他方案定义?
- Gussian Distribution?
