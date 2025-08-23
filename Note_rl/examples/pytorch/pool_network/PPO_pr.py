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


class Controller(nn.Module):
    def __init__(self, hidden=32, temp=10.0):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.max_w = None
        self.temp = temp

    def forward(self, features):
        # features: (1,2) tensor
        x = F.relu(self.fc1(features))
        alpha = torch.sigmoid(self.fc2(x))  # (1,1)
        if self.max_w is None:
            raise RuntimeError("Controller.max_w is not set")
        w = alpha * float(self.max_w)
        return torch.squeeze(w, dim=-1)


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


class PPO_(RL_pytorch):
    def __init__(self, state_dim, hidden_dim, action_dim, clip_eps, alpha, processes, temp=10.0, gamma=0.98, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.actor = Actor(state_dim, hidden_dim, action_dim).to(self.device)
        self.actor_old = Actor(state_dim, hidden_dim, action_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, hidden_dim).to(self.device)
        self.controller = Controller(hidden=32, temp=temp).to(self.device)

        self.clip_eps = clip_eps
        self.alpha = alpha
        self.temp = temp
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
        ratio_score = torch.sum(torch.abs(self.prioritized_replay.ratio_list[p]-1.0))
        td_score = torch.sum(self.prioritized_replay.TD_list[p])
        scores = self.lambda_ * self.prioritized_replay.TD_list[p] + (1.0-self.lambda_) * torch.abs(self.prioritized_replay.ratio_list[p] - 1.0)
        weights = torch.pow(scores + 1e-7, self.alpha)
        p = weights / (torch.sum(weights))
        ess = 1.0 / (torch.sum(p * p))
        features = torch.reshape(torch.stack([ratio_score, td_score, ess, self.prioritized_replay.ratio.shape[0]]), (1,4)).to(self.device)
        mn = torch.min(features)
        mx = torch.max(features)
        features = (features - mn) / (mx - mn + 1e-8)
        self.controller.max_w = self.prioritized_replay.ratio_list[p].shape[0]
        w = self.controller(features)
        return w

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

        ratio_score = torch.sum(torch.abs(self.prioritized_replay.ratio-1.0))
        td_score = torch.sum(self.prioritized_replay.TD)
        score = ratio_score + td_score
        scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * torch.abs(self.prioritized_replay.ratio - 1.0)
        weights = torch.pow(scores + 1e-7, self.alpha)
        p = weights / (torch.sum(weights))
        ess = 1.0 / (torch.sum(p * p))
        features = torch.reshape(torch.stack([ratio_score, td_score, ess, self.prioritized_replay.ratio.shape[0]]), (1,4))
        mn = torch.min(features)
        mx = torch.max(features)
        features = (features - mn) / (mx - mn + 1e-8)

        self.controller.max_w = self.prioritized_replay.ratio.shape[0]
        w = self.controller(features)  # scalar tensor

        idx = torch.arange(self.prioritized_replay.ratio.shape[0], dtype=w.dtype, device=self.device)
        m = torch.sigmoid((idx - w) / self.temp)  # (N,)
        controller_loss = -torch.mean(m * score)  # scalar

        loss = torch.mean(clip_loss) + torch.mean(TD ** 2) + controller_loss
        self.prioritized_replay.update(TD,ratio)
        return loss

    def update_param(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        return
