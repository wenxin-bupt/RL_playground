import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 模拟一个简单的训练过程
def simulate_training():
    # 初始化参数
    theta = torch.tensor([-0.5, 0.5], requires_grad=True)
    learning_rate = 0.1
    
    probs_history = []
    loss_history = []
    log_prob_history = []
    
    # 模拟训练过程
    for _ in range(50):
        # 计算动作概率
        probs = F.softmax(theta, dim=0)
        probs_history.append(probs.detach().numpy())
        
        # 假设选择了动作1，回报为1
        log_prob = torch.log(probs[1])
        G = 1.0
        loss = -log_prob * G
        
        # 记录历史
        loss_history.append(loss.item())
        log_prob_history.append(log_prob.item())
        
        # 计算梯度并更新
        loss.backward()
        with torch.no_grad():
            theta -= learning_rate * theta.grad
            theta.grad.zero_()
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 概率变化
    plt.subplot(131)
    steps = range(len(probs_history))
    plt.plot(steps, [p[0] for p in probs_history], label='Action 0')
    plt.plot(steps, [p[1] for p in probs_history], label='Action 1')
    plt.xlabel('Training Steps')
    plt.ylabel('Probability')
    plt.title('Action Probabilities Over Time')
    plt.legend()
    plt.grid(True)
    
    # Loss变化
    plt.subplot(132)
    plt.plot(steps, loss_history, 'r-', label='Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.title('Loss Over Time\n(-log_prob * G)')
    plt.grid(True)
    plt.legend()
    
    # Log概率变化
    plt.subplot(133)
    plt.plot(steps, log_prob_history, 'g-', label='Log Prob')
    plt.xlabel('Training Steps')
    plt.ylabel('Log Probability')
    plt.title('Log Probability Over Time')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

simulate_training()

# 补充：展示具体的数值变化
def show_numerical_changes():
    # 创建一个表格来显示训练过程中的具体值
    prob = 0.3  # 初始概率
    steps = 5
    
    print("Step | Probability | Log Prob | Loss (-log_prob * G) | Gradient")
    print("-" * 60)
    
    for i in range(steps):
        log_prob = np.log(prob)
        G = 1.0
        loss = -log_prob * G
        grad = -G * (1/prob)  # 梯度
        
        print(f"{i:4d} | {prob:11.3f} | {log_prob:8.3f} | {loss:18.3f} | {grad:8.3f}")
        
        # 更新概率（简化的更新）
        prob += 0.1
