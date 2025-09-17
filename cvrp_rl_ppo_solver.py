import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ================== 환경 정의 ==================
class CVRPEnv:
    def __init__(self, coords, demands, capacity=25):
        self.coords = np.array(coords, dtype=np.float32)
        self.demands = np.array(demands, dtype=np.int32)
        self.capacity = capacity
        self.N = len(demands)
        self.max_steps = 2*self.N

    def reset(self):
        self.visited = np.zeros(self.N, dtype=bool)
        self.remaining_capacity = self.capacity
        self.current_pos = 0
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        flat_coords = self.coords.flatten()
        obs = np.concatenate([
            flat_coords,
            self.demands,
            self.visited.astype(np.float32),
            self.coords[self.current_pos],
            [self.remaining_capacity]
        ]).astype(np.float32)
        return obs

    def get_mask(self):
        # capacity 다 떨어졌고 현재 depot이 아니라면 -> depot만 선택 가능
        if self.remaining_capacity <= 0 and self.current_pos != 0:
            return [True] + [False]*self.N

        mask = []
        for i in range(self.N+1):
            if i == 0:  # depot
                if self.current_pos == 0:
                    mask.append(False)  # depot에 있을 땐 다시 depot 선택 불가
                else:
                    mask.append(True)   # depot 복귀 가능
            else:
                idx = i-1
                if self.visited[idx] or self.demands[idx] > self.remaining_capacity:
                    mask.append(False)
                else:
                    mask.append(True)
        return mask
    def step(self, action):
        done = False
        reward = 0.0

        if self.remaining_capacity <= 0 and action != 0:
            return self._get_obs(), -1e6, True, {}

        if action == 0:  # depot
            dist = np.linalg.norm(self.coords[self.current_pos]-self.coords[0])
            reward = -dist
            self.current_pos = 0
            self.remaining_capacity = self.capacity
        else:  # 고객
            idx = action-1
            if self.visited[idx] or self.demands[idx] > self.remaining_capacity:
                return self._get_obs(), -1e6, True, {}
            dist = np.linalg.norm(self.coords[self.current_pos]-self.coords[idx+1])
            reward = -dist
            self.current_pos = idx+1
            self.remaining_capacity -= self.demands[idx]
            self.visited[idx] = True

        if all(self.visited) and self.current_pos == 0:
            done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

# ================== PPO 네트워크 ==================
class Policy(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.obs_dim = 4*N + 5
        self.enc = nn.Sequential(
            nn.Linear(self.obs_dim,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU()
        )
        self.pi = nn.Linear(256,N+1)
        self.v = nn.Linear(256,1)
    def forward(self,x):
        h = self.enc(x)
        return self.pi(h), self.v(h)

# ================== PPO 학습 ==================
def ppo_train(env, policy, episodes=200, gamma=0.99, clip=0.2, lr=1e-4, T=200):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    all_rewards = []
    for ep in range(episodes):
        obs = env.reset()
        log_probs, values, rewards = [], [], []
        total_distance = 0
        for t in range(T):
            x = torch.tensor(obs).float().unsqueeze(0)
            logits, value = policy(x)
            logits = logits.squeeze(0)
            mask = env.get_mask()
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            logits_masked = logits.clone()
            logits_masked[~mask_tensor] = -1e9
            probs = F.softmax(logits_masked, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action))

            prev_pos = env.current_pos
            next_obs, reward, done, _ = env.step(action)

            if action == 0:
                step_distance = np.linalg.norm(env.coords[prev_pos]-env.coords[0])
            else:
                idx = action-1
                step_distance = np.linalg.norm(env.coords[prev_pos]-env.coords[idx+1])
            total_distance += step_distance

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            obs = next_obs
            if done:
                break

        R, returns = 0, []
        for r in rewards[::-1]:
            R = r + gamma*R
            returns.insert(0,R)
        returns = torch.tensor(returns).float()
        values = torch.cat(values).squeeze()
        log_probs = torch.stack(log_probs)
        advantage = returns - values.detach()
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-clip, 1+clip) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_rewards.append(sum(rewards))

        print(f"Episode {ep+1} done, total reward {sum(rewards):.2f}, total distance {total_distance:.2f}", flush=True)

    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("C:/Users/pwh/Downloads/open/reward_curve.png")
    return policy

# ================== 2-opt ==================
def two_opt(route, coords):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1,len(best)-2):
            for j in range(i+1,len(best)):
                if j-i==1: continue
                new_route = best[:]
                new_route[i:j] = best[j-1:i-1:-1]
                if route_length(new_route,coords)<route_length(best,coords):
                    best = new_route
                    improved = True
        route = best
    return best

def route_length(route,coords):
    d=0
    for i in range(len(route)-1):
        d += np.linalg.norm(coords[route[i]]-coords[route[i+1]])
    return d

# ================== 실행 ==================
data = pd.read_csv("C:/Users/pwh/Downloads/open/data.csv")
coords = [(0,0)]+list(zip(data["x"], data["y"]))
demands = data["demand"].tolist()

N = len(demands)
policy = Policy(N)
env = CVRPEnv(coords, demands, capacity=25)
policy = ppo_train(env, policy, episodes=2000)

# 경로 추론
obs = env.reset()
route = [0]
while True:
    x = torch.tensor(obs).float().unsqueeze(0)
    logits, _ = policy(x)
    logits = logits.squeeze(0)
    mask = env.get_mask()
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    logits_masked = logits.clone()
    logits_masked[~mask_tensor] = -1e9
    probs = F.softmax(logits_masked, dim=-1).detach().numpy()
    action = np.argmax(probs)
    obs, reward, done, _ = env.step(action)
    route.append(env.current_pos)
    if done:
        break

route = two_opt(route, np.array(coords))

print("Final route:", route)

out = pd.DataFrame({"trip":[1]*len(route),"order":list(range(len(route))),"point_id":route})
out.to_csv("C:/Users/pwh/Downloads/open/rl_routes.csv",index=False)
print("Saved rl_routes.csv and reward_curve.png")
