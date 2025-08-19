import torch
from Note_rl.RL_pytorch import RL_pytorch
import torch.nn as nn
import torch.nn.functional as F
import gym


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=-1)
        return probs  # (batch, action_dim)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc2(x)
        return v  # (batch, 1)


class PPO(RL_pytorch):
    def __init__(self, state_dim, hidden_dim, action_dim, clip_eps, alpha, processes, gamma=0.98, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.actor = Actor(state_dim, hidden_dim, action_dim).to(self.device)
        self.actor_old = Actor(state_dim, hidden_dim, action_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, hidden_dim).to(self.device)

        self.clip_eps = clip_eps
        self.alpha = alpha
        self.gamma = gamma

        self.param = [self.actor.parameters(), self.critic.parameters(), self.controller.parameters()]

        # env
        self.env = [gym.make('CartPole-v0') for _ in range(processes)]

    def action(self, s):
        # s: torch.Tensor on device, shape (batch, state_dim) or (state_dim,)
        with torch.no_grad():
            probs = self.actor_old(s.to(self.device))
        return probs

    def window_size_fn(self, p):
        return self.adjust_window_size(p)

    def __call__(self, s, a, next_s, r, d):
        s = s.to(self.device)
        next_s = next_s.to(self.device)
        a = a.to(self.device).long().view(-1, 1)          # (batch,1)
        r = r.to(self.device).float().view(-1, 1)        # (batch,1)
        d = d.to(self.device).float().view(-1, 1)        # (batch,1)

        # actor probabilities
        probs = self.actor(s)            # (batch, action_dim)
        probs_old = self.actor_old(s)    # (batch, action_dim)

        # gather action probs
        action_prob = probs.gather(1, a).squeeze(1)      # (batch,)
        action_prob_old = probs_old.gather(1, a).squeeze(1)  # (batch,)

        # ratio
        ratio = action_prob / (action_prob_old + 1e-8)   # (batch,)

        # value and target
        value = self.critic(s).squeeze(1)             # (batch,)
        value_next = self.critic(next_s).squeeze(1)   # (batch,)
        value_tar = r.squeeze(1) + self.gamma * value_next * (1.0 - d.squeeze(1))
        TD = value_tar - value                        # (batch,)

        # surrogate losses
        sur1 = ratio * TD
        sur2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * TD
        # elementwise minimum
        clip_loss = -torch.min(sur1, sur2)            # (batch,)

        entropy = action_prob * torch.log(action_prob + 1e-8)  # (batch,)
        clip_loss = clip_loss - self.alpha * entropy
        
        loss = torch.mean(clip_loss) + torch.mean(TD ** 2)
        self.prioritized_replay.update(TD,ratio)
        return loss

    def update_param(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        return
