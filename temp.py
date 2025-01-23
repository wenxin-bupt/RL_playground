import gym

env = gym.make('CartPole-v0')
env.seed(0)

# 打印多次重置后的初始状态
for i in range(5):
    state = env.reset()
    print(f"Reset {i+1}: {state}")