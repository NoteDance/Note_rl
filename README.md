# Introduction:
This libaray allows you to easily train agents built with Keras or PyTorch using reinforcement learning. You just need to have your agent class inherit from the RL or RL_pytorch class, and you can easily train your agent built with Keras or PyTorch. You can learn how to build an agent from the examples [here](https://github.com/NoteDance/Reinforcement-Learning/tree/main/Note_rl/examples). The README shows how to train, save, and restore agent built with Keras or PyTorch.

# Installation:
To use this library, you need to download it and then unzip it to the site-packages folder of your Python environment.

**dependent packages**:

tensorflow>=2.16.1

pytorch>=2.3.1

gym<=0.25.2

matplotlib>=3.8.4

**python requirement**:

python>=3.10

# RL.set:

Configures the core hyperparameters and features of the reinforcement learning agent. This method must be called before training to specify the algorithm behavior, replay buffer settings, and advanced options (e.g., PPO, HER, Prioritized Replay).

**Parameters:**

| Parameter             | Type                  | Default    | Description |
|-----------------------|-----------------------|------------|-------------|
| `policy`              | Policy object or list | `None`     | Exploration policy (e.g., `rl.EpsGreedyQPolicy`, `rl.SoftmaxPolicy`). Can be a list for multi-agent setups. |
| `noise`               | Noise object or list  | `None`     | Action noise for continuous control (e.g., Ornstein-Uhlenbeck noise). Can be a list for multi-agent. |
| `pool_size`           | `int`                 | `None`     | Maximum size of the replay buffer (experience pool). |
| `batch`               | `int`                 | `None`     | Mini-batch size for training updates. |
| `num_updates`         | `int` or `None`       | `None`     | Number of training updates per episode (if `None`, trains on full buffer). |
| `num_steps`           | `int` or `None`       | `None`     | Number of environment steps per stored transition (for n-step returns). |
| `update_batches`      | `int`                 | `None`     | Frequency of parameter updates when using pooled networks (e.g., update target every N batches). |
| `update_steps`        | `int` or `None`       | `None`     | Frequency of parameter updates in steps (alternative to `update_batches`). |
| `trial_count`         | `int` or `None`       | `None`     | Number of recent episodes to average for early stopping or best-model saving. |
| `criterion`           | `float` or `None`      | `None`     | Reward threshold for early stopping (stops when average reward over `trial_count` meets/exceeds this). |
| `PPO`                 | `bool`                | `False`    | Enable Proximal Policy Optimization mode (uses ratio-based prioritization). |
| `HER`                 | `bool`                | `False`    | Enable Hindsight Experience Replay. |
| `TRL`                 | `bool`                | `False`    | Enable Trajectory-based Reinforcement Learning (custom triplet sampling). |
| `MARL`                | `bool`                | `False`    | Enable Multi-Agent Reinforcement Learning mode. |
| `PR`                  | `bool`                | `False`    | Enable Prioritized Experience Replay. |
| `IRL`                 | `bool`                | `False`    | Enable Inverse Reinforcement Learning mode. |
| `initial_ratio`       | `float`               | `1.0`      | Initial importance-sampling ratio for PPO prioritized replay. |
| `initial_TD`          | `float`               | `7.0`      | Initial TD-error value for prioritized replay initialization. |
| `lambda_`             | `float`               | `0.5`      | Weighting factor for combining TD-error and ratio deviation in PPO prioritization. |
| `alpha`               | `float`               | `0.7`      | Prioritization exponent for experience sampling. |

**Returns:**  
None (configures the agent in-place).

**Example:**
```python
rl_agent.set(
    policy=rl.EpsGreedyQPolicy(eps=0.1),
    noise=None,
    pool_size=100000,
    batch=64,
    num_updates=4,
    PPO=True,
    PR=True,
    trial_count=100,
    criterion=500.0
)
```

# RL.train

Trains the agent with highly configurable single- or multi-process experience collection and training loops.

**Parameters:**

| Parameter                       | Type                  | Default   | Description |
|---------------------------------|-----------------------|-----------|-------------|
| `train_loss`                    | `tf.keras.metrics.Metric` | **Required** | Metric (e.g., `tf.keras.metrics.Mean()`) to track training loss. |
| `optimizer`                     | Optimizer or list     | `None`    | Optimizer(s). Uses previously set optimizer if `None`. |
| `episodes`                      | `int` or `None`       | `None`    | Number of episodes to train. If `None`, trains indefinitely until stopped. |
| `pool_network`                  | `bool`                | `True`    | Use multiple parallel environments for faster data collection. |
| `parallel_store_and_training`   | `bool`                | `True`    | Run experience collection and training in parallel processes (non-blocking). |
| `parallel_training_and_save`    | `bool`                | `False`   | Run checkpoint saving in a separate process (non-blocking). |
| `processes`                     | `int` or `None`       | `None`    | Number of parallel environment processes (typically number of CPU cores). |
| `num_store`                     | `int`                 | `1`       | Number of collection cycles per training update in parallel mode. |
| `processes_her`                 | `int` or `None`       | `None`    | Parallel processes for Hindsight Experience Replay (HER) sampling. |
| `processes_pr`                  | `int` or `None`       | `None`    | Parallel processes for prioritized replay sampling. |
| `window_size`                   | `int`, `float`, callable, or `None` | `None` | Buffer trimming: keep recent fraction or fixed number of transitions (or callable returning size). |
| `clearing_freq`                 | `int` or `None`       | `None`    | Frequency to clear old transitions from per-process buffers. |
| `window_size_`                  | `int` or `None`       | `None`    | General fallback window size. |
| `window_size_ppo`               | `int` or `None`       | `None`    | Window size specific to PPO prioritization. |
| `window_size_pr`                | `int` or `None`       | `None`    | Window size specific to standard prioritized replay. |
| `jit_compile`                   | `bool`                | `True`    | Enable XLA/JIT compilation for faster training steps. |
| `random`                        | `bool`                | `False`   | Randomly distribute stored transitions across process buffers (load balancing). |
| `save_data`                     | `bool`                | `True`    | Include replay buffer data when saving the model. |
| `callbacks`                     | `list` or `None`      | `None`    | List of Keras-style callbacks (e.g., logging, early stopping). |
| `p`                             | `int` or `None`       | `None`    | Print progress every `p` episodes (default: every ~10% of total). |

**Returns:**  
None. Prints episode progress, rewards (current + rolling average if `trial_count` set), loss, and total training time.

**Example:**
```python
rl_agent.train(
    train_loss=train_loss,
    episodes=5000,
    pool_network=True,
    parallel_store_and_training=True,
    parallel_training_and_save=True,
    processes=12,
    jit_compile=True,
    p=50
)
```

# RL.distributed_training

Distributed training across GPUs/workers using TensorFlow distribution strategies (`MirroredStrategy`, `MultiWorkerMirroredStrategy`, `ParameterServerStrategy`). Supports all features of `train()`.

**Parameters:**  
All parameters from `train()` **plus**:

| Parameter                       | Type                  | Default   | Description |
|---------------------------------|-----------------------|-----------|-------------|
| `optimizer`                     | Optimizer or list     | `None`    | Same as `train()`. |
| `strategy`                      | `tf.distribute.Strategy` | **Required** | Distribution strategy (e.g., `tf.distribute.MirroredStrategy()`). |
| `episodes`                      | `int` or `None`       | `None`    | Total episodes (used primarily with `MirroredStrategy`). |
| `num_episodes`                  | `int` or `None`       | `None`    | Alternative episode count (used with `MultiWorkerMirroredStrategy`/`ParameterServerStrategy`). |

**Returns:**  
None. Training runs across devices/workers using the specified strategy.

**Example (MirroredStrategy):**
```python
strategy = tf.distribute.MirroredStrategy()

rl_agent.distributed_training(
    train_loss=train_loss,
    strategy=strategy,
    episodes=10000,
    pool_network=True,
    parallel_store_and_training=True,
    processes=8,
    jit_compile=True
)
```

**Example (ParameterServerStrategy):**
```python
strategy = tf.distribute.experimental.ParameterServerStrategy()

rl_agent.distributed_training(
    train_loss=train_loss,
    strategy=strategy,
    num_episodes=20000,
    processes=16
)
```

# Advanced Adaptive Hyperparameter Adjustment

The `RL` class includes powerful adaptive mechanisms to dynamically tune hyperparameters during training based on **Effective Sample Size (ESS)** from prioritized weights or **gradient noise**. These methods help combat issues like weight collapse in prioritized replay, improve sample efficiency, and stabilize training. They are particularly useful with `PR=True` (Prioritized Replay) or `PPO=True`.

These functions are typically called automatically during training if you set `self.adjust_func` (e.g., via a lambda), or you can invoke them manually after episodes.

## `adjust_window_size(p=None, scale=1.0, ema=None)`

Dynamically computes a window size for trimming the replay buffer to maintain healthy ESS (effective sample size). Low ESS indicates collapsed priorities; trimming removes low-priority old experiences.

**Parameters:**

| Parameter | Type              | Default | Description |
|---------|-------------------|---------|-------------|
| `p`     | `int` or `None`   | `None`  | Process index (for parallel/multi-environment mode). If `None`, uses global buffer. |
| `scale` | `float`           | `1.0`   | Scaling factor for aggressiveness of trimming (higher = more aggressive removal). |
| `ema`   | `float` or `None` | `None`  | Pre-computed EMA of ESS (for custom smoothing). If `None`, computed internally. |

**Returns:**  
`int` – The number of oldest experiences to remove (clamped to valid range).

**Usage Note:**  
Used internally when `window_size_func` is set or during prioritized buffer maintenance.

**Example:**

https://github.com/NoteDance/Note_rl/blob/main/Note_rl/examples/keras/pool_network/PPO_pr.py
https://github.com/NoteDance/Note_rl/blob/main/Note_rl/examples/pytorch/pool_network/PPO_pr.py

## `adabatch(num_samples, target_noise=1e-3, smooth=0.2, batch_params=None, alpha_params=None, eps_params=None, tau_params=None, gamma_params=None, clip_params=None, beta_params=None, jit_compile=True)`

Gradient-noise-based adaptation (AdaBatch-style). Estimates variance of gradients across multiple mini-batches and adjusts hyperparameters to reach a target noise level.

**Parameters:**

| Parameter       | Type     | Default   | Description |
|-----------------|----------|-----------|-------------|
| `num_samples`   | `int`    | **Required** | Number of mini-batches to sample for variance estimation. |
| `target_noise`  | `float`  | `1e-3`    | Desired gradient variance target. |
| `smooth`        | `float`  | `0.2`     | EMA smoothing for noise estimate. |
| `batch_params`, `alpha_params`, etc. | `dict` or `None` | `None` | Same as in `adjust_batch_size`. |
| `jit_compile`   | `bool`   | `True`    | Use XLA compilation for faster gradient estimation. |

## `adjust(target_ess=None, target_noise=None, num_samples=None, smooth=0.2, batch_params=None, alpha_params=None, ..., jit_compile=True)`

Unified entry point for adaptation. Chooses between ESS-based (`target_ess`) and noise-based (`target_noise`) adjustment.

**Parameters:**

- If `target_ess` is provided → calls `adjust_batch_size`.
- If `target_noise` is provided → calls `adabatch` (requires `num_samples`).

All other parameters are passed through to the chosen method.

**Example:**

https://github.com/NoteDance/Note_rl/blob/main/Note_rl/examples/keras/pool_network/PPO_pr.py
https://github.com/NoteDance/Note_rl/blob/main/Note_rl/examples/pytorch/pool_network/PPO_pr.py

# Single-Machine Training

## TensorFlow / Note / Keras

```python
import tensorflow as tf
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.keras.DQN import DQN

# Basic DQN
model = DQN(state_dim=4, hidden_size=128, action_dim=2)
model.set(
    policy=EpsGreedyQPolicy(eps=0.01),
    pool_size=10000,
    batch=64,
    update_steps=10
)

optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')

model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=100,
    pool_network=False
)
```

```python
# DQN with early stopping
model.set(
    policy=EpsGreedyQPolicy(eps=0.01),
    pool_size=10000,
    batch=64,
    update_steps=10,
    trial_count=10,
    criterion=200.0
)

model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=100,
    pool_network=False
)
```

```python
# Periodic checkpointing (max 2 files)
model.path = 'model.dat'
model.save_freq = 10
model.max_save_files = 2

model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=100,
    pool_network=False
)
```

```python
# Save parameters only
model.path = 'param.dat'
model.save_freq = 10
model.max_save_files = 2
model.save_param_only = True

model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=100,
    pool_network=False
)
```

```python
# Save best model only
model.path = 'model.dat'
model.save_best_only = True

model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=100,
    pool_network=False
)
```

```python
# PPO example
from Note_rl.examples.keras.PPO import PPO

model = PPO(state_dim=4, hidden_size=128, action_dim=2, clip=0.7, entropy_coef=0.7)
model.set(
    policy=SoftmaxPolicy(),
    pool_size=10000,
    batch=64,
    update_steps=1000,
    PPO=True
)

optimizer = [
    tf.keras.optimizers.Adam(1e-4),   # policy optimizer
    tf.keras.optimizers.Adam(5e-3)    # value optimizer
]

model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=100,
    pool_network=False
)
```

```python
# HER (DDPG + HER)
from Note_rl.examples.keras.DDPG_HER import DDPG

model = DDPG(hidden_size=128, tau=0.1, gamma=0.98, lr=0.005)
model.set(
    noise=GaussianWhiteNoiseProcess(),
    pool_size=10000,
    batch=256,
    trial_count=10,
    criterion=-5.0,
    HER=True
)

optimizer = [
    tf.keras.optimizers.Adam(),  # actor
    tf.keras.optimizers.Adam()   # critic
]

model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=2000,
    pool_network=False
)
```

```python
# Multi-Agent (MADDPG)
from Note_rl.examples.keras.MADDPG import DDPG

model = DDPG(hidden_size=128, tau=0.1, gamma=0.98, lr=0.005)
model.set(
    policy=SoftmaxPolicy(),
    pool_size=3000,
    batch=32,
    trial_count=10,
    MARL=True
)

optimizer = [
    tf.keras.optimizers.Adam(),  # actor
    tf.keras.optimizers.Adam()   # critic
]

model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=100,
    pool_network=False
)
```

```python
# Pool Network (parallel environments)
from Note_rl.examples.keras.pool_network.DQN import DQN

model = DQN(state_dim=4, hidden_size=128, action_dim=2, processes=7)
model.set(
    policy=EpsGreedyQPolicy(eps=0.01),
    pool_size=10000,
    batch=64,
    update_batches=17
)

model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=100,
    pool_network=True,
    processes=7
)
```

```python
# Pool Network + Parallel HER sampling
from Note_rl.examples.keras.pool_network.DDPG_HER import DDPG

model = DDPG(hidden_size=128, tau=0.1, gamma=0.98, lr=0.005, processes=7)
model.set(
    noise=GaussianWhiteNoiseProcess(),
    pool_size=10000,
    batch=256,
    trial_count=10,
    HER=True
)

model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=2000,
    pool_network=True,
    processes=7,
    processes_her=4
)
```

```python
# Pool Network + Parallel Prioritized Replay sampling
from Note_rl.examples.keras.pool_network.DQN_pr import DQN

model = DQN(state_dim=4, hidden_size=128, action_dim=2, processes=7)
model.set(
    policy=EpsGreedyQPolicy(eps=0.01),
    pool_size=10000,
    batch=64,
    update_batches=17,
    PR=True
)

model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=100,
    pool_network=True,
    processes=7,
    processes_pr=4
)
```

## PyTorch

```python
import torch
from Note.RL import rl
from Note_rl.examples.pytorch.DQN import DQN

# Basic DQN
model = DQN(state_dim=4, hidden_size=128, action_dim=2)
model.set(
    policy=EpsGreedyQPolicy(eps=0.01),
    pool_size=10000,
    batch=64,
    update_steps=10
)

optimizer = torch.optim.Adam(model.param)

model.train(
    optimizer=optimizer,
    episodes=100,
    pool_network=False
)
```

```python
# With early stopping
model.set(
    policy=EpsGreedyQPolicy(eps=0.01),
    pool_size=10000,
    batch=64,
    update_steps=10,
    trial_count=10,
    criterion=200.0
)

model.train(optimizer=optimizer, episodes=100, pool_network=False)
```

```python
# Prioritized Replay
model.set(
    policy=EpsGreedyQPolicy(eps=0.01),
    pool_size=10000,
    batch=64,
    update_steps=10,
    PR=True,
    initial_TD=7.0,
    alpha=0.7
)

model.train(optimizer=optimizer, episodes=100, pool_network=False)
```

```python
# HER (DDPG + HER)
from Note_rl.examples.pytorch.DDPG_HER import DDPG

model = DDPG(hidden_size=128, tau=0.1, gamma=0.98, lr=0.005)
model.set(
    noise=GaussianWhiteNoiseProcess(),
    pool_size=10000,
    batch=256,
    trial_count=10,
    criterion=-5.0,
    HER=True
)

optimizer = [
    torch.optim.Adam(model.param[0]),  # actor
    torch.optim.Adam(model.param[1])   # critic
]

model.train(optimizer=optimizer, episodes=2000, pool_network=False)
```

```python
# Multi-Agent (MADDPG)
from Note_rl.examples.pytorch.MADDPG import DDPG

model = DDPG(hidden_size=128, tau=0.1, gamma=0.98, lr=0.005)
model.set(
    policy=SoftmaxPolicy(),
    pool_size=3000,
    batch=32,
    trial_count=10,
    MARL=True
)

optimizer = [
    torch.optim.Adam(model.param[0]),
    torch.optim.Adam(model.param[1])
]

model.train(optimizer=optimizer, episodes=100, pool_network=False)
```

```python
# Pool Network (parallel environments)
from Note_rl.examples.pytorch.pool_network.DQN import DQN

model = DQN(state_dim=4, hidden_size=128, action_dim=2, processes=7)
model.set(
    policy=EpsGreedyQPolicy(eps=0.01),
    pool_size=10000,
    batch=64,
    update_batches=17
)

model.train(
    optimizer=optimizer,
    episodes=100,
    pool_network=True,
    processes=7
)
```

```python
# Pool Network + Parallel HER sampling
from Note_rl.examples.pytorch.pool_network.DDPG_HER import DDPG

model = DDPG(hidden_size=128, tau=0.1, gamma=0.98, lr=0.005, processes=7)
model.set(
    noise=rl.GaussianWhiteNoiseProcess(),
    pool_size=10000,
    batch=256,
    trial_count=10,
    HER=True
)

optimizer = [
    torch.optim.Adam(model.param[0]),
    torch.optim.Adam(model.param[1])
]

model.train(
    optimizer=optimizer,
    episodes=2000,
    pool_network=True,
    processes=7,
    processes_her=4
)
```

# Distributed Training (TensorFlow)

## MirroredStrategy (Multi-GPU, single machine)

```python
import tensorflow as tf
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.keras.DQN import DQN

strategy = tf.distribute.MirroredStrategy()
batch_per_replica = 64
global_batch = batch_per_replica * strategy.num_replicas_in_sync

with strategy.scope():
    model = DQN(state_dim=4, hidden_size=128, action_dim=2)
    optimizer = tf.keras.optimizers.Adam()

model.set(
    policy=EpsGreedyQPolicy(eps=0.01),
    pool_size=10000,
    batch=global_batch,
    update_steps=10
)

model.distributed_training(
    optimizer=optimizer,
    strategy=strategy,
    episodes=100,
    pool_network=False
)
```

```python
# PPO with MirroredStrategy
from Note_rl.examples.keras.PPO import PPO

with strategy.scope():
    model = PPO(state_dim=4, hidden_size=128, action_dim=2, clip=0.7, entropy_coef=0.7)
    optimizer = [
        tf.keras.optimizers.Adam(1e-4),
        tf.keras.optimizers.Adam(5e-3)
    ]

model.set(
    policy=SoftmaxPolicy(),
    pool_size=10000,
    batch=global_batch,
    update_steps=1000,
    PPO=True
)

model.distributed_training(
    optimizer=optimizer,
    strategy=strategy,
    episodes=100,
    pool_network=False
)
```

```python
# Pool Network + MirroredStrategy
from Note_rl.examples.keras.pool_network.DQN import DQN

with strategy.scope():
    model = DQN(state_dim=4, hidden_size=128, action_dim=2, processes=7)
    optimizer = tf.keras.optimizers.Adam()

model.set(
    policy=EpsGreedyQPolicy(eps=0.01),
    pool_size=10000,
    batch=global_batch,
    update_batches=17
)

model.distributed_training(
    optimizer=optimizer,
    strategy=strategy,
    episodes=100,
    pool_network=True,
    processes=7
)
```

## MultiWorkerMirroredStrategy (Multi-machine)

```python
import tensorflow as tf
import os
import sys

# Disable GPU on workers if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
    sys.path.insert(0, '.')

tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}
os.environ['TF_CONFIG'] = json.dumps(tf_config)

strategy = tf.distribute.MultiWorkerMirroredStrategy()
per_worker_batch = 64
num_workers = len(tf_config['cluster']['worker'])
global_batch = per_worker_batch * num_workers

with strategy.scope():
    model = DQN(state_dim=4, hidden_size=128, action_dim=2)
    optimizer = tf.keras.optimizers.Adam()

model.set(
    policy=EpsGreedyQPolicy(eps=0.01),
    pool_size=10000,
    batch=global_batch,
    update_batches=17
)

model.distributed_training(
    optimizer=optimizer,
    strategy=strategy,
    num_episodes=100,           # use num_episodes for multi-worker
    pool_network=True,
    processes=7
)
```

**Visualization & Saving** (common to all examples):

```python
# Plot results
model.visualize_loss()
model.visualize_reward()
model.visualize_reward_loss()

# Animate trained agent
model.animate_agent(max_steps=200)

# Manual save
model.save_param('param.dat')
model.save('model.dat')
```

# Building a Custom Agent by Extending the RL Base Class

This example shows how to create a fully functional reinforcement learning agent by inheriting from the provided `RL` base class. The design leverages two key components from the `Note` framework:

- `nn.Model`: A lightweight neural network base class that simplifies layer definition and parameter management.
- `nn.RL`: The powerful reinforcement learning base class that handles replay buffers, parallel environments (Pool Network), prioritized replay, training loops, saving/loading, visualization, and more.

By combining these, you can build complex agents (DQN, DDPG, PPO, etc.) with minimal boilerplate code.

## Step 1: Define the Neural Network (Q-Network)

## Step 2: Create the DQN Agent by Inheriting from `RL`

```python
import gym
import tensorflow as tf

class DQN(nn.RL):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        
        # Main and target Q-networks
        self.q_net = Qnet(state_dim, hidden_dim, action_dim)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim)
        
        # Use main network parameters for optimization
        self.param = self.q_net.param
        
        # Built-in environment (CartPole-v0)
        self.env = gym.make('CartPole-v0').env  # .env removes time limit warning

    # Forward pass: returns Q-values for given states
    def action(self, s):
        return self.q_net(s)

    # Loss computation (called during training)
    def __call__(self, s, a, next_s, r, d):
        # Gather Q-values for taken actions
        a = tf.expand_dims(a, axis=1)
        current_q = tf.gather(self.q_net(s), a, axis=1, batch_dims=1)
        
        # Double DQN: use online net to select, target net to evaluate
        next_q_online = self.q_net(next_s)
        best_action = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
        best_action = tf.expand_dims(best_action, axis=1)
        next_q_target = tf.gather(self.target_q_net(next_s), best_action, axis=1, batch_dims=1)
        
        # TD target
        target = r + 0.99 * next_q_target * (1.0 - d)
        
        # Huber loss for stability (optional, or use MSE)
        td_error = current_q - tf.stop_gradient(target)
        loss = tf.reduce_mean(tf.square(td_error))
        
        return loss

    # Soft target network update (polyak averaging) - optional but recommended
    def update_param(self):
        tau = 0.005  # Soft update rate
        for target_param, param in zip(self.target_q_net.param, self.q_net.param):
            target_param.assign(tau * param + (1.0 - tau) * target_param)
```

### Key Methods Explained

- `action(self, s)`: Returns Q-values. Used by the base `RL` class during action selection (combined with policy/noise).
- `__call__(self, s, a, next_s, r, d)`: Computes the training loss. Called automatically during gradient updates.
- `update_param(self)`: Called periodically (via `update_steps` or `update_batches`) to synchronize the target network. Override for soft/hard updates.

## Step 3: Train the Agent

```python
import tensorflow as tf
from Note_rl.policy import EpsGreedyQPolicy

# Instantiate and configure
model = DQN(state_dim=4, hidden_dim=128, action_dim=2)

model.set(
    policy=EpsGreedyQPolicy(eps=0.05),   # Exploration policy
    pool_size=100000,                       # Replay buffer size
    batch=64,                               # Mini-batch size
    update_steps=10                         # Update target every 10 steps
)

# Optimizer and loss tracker
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_loss = tf.keras.metrics.Mean(name='train_loss')

# Train
model.train(
    train_loss=train_loss,
    optimizer=optimizer,
    episodes=500,
    pool_network=False  # Set True + processes=N for parallel environments
)
```

### Optional Enhancements

```python
# Early stopping when average reward over last 50 episodes >= 195
model.set(trial_count=50, criterion=195.0)

# Periodic checkpointing (keep max 3 files)
model.path = 'dqn_cartpole.dat'
model.save_freq = 20
model.max_save_files = 3
```

## Supporting Hindsight Experience Replay (HER)

To use **HER**, define a custom reward function that evaluates achievement of arbitrary goals:

```python
class DDPG_HER(nn.RL):
    def __init__(...):
        ...
        self.env = gym.make('FetchReach-v1')

    def reward_done_func(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        done = distance < 0.05
        reward = -1.0 if not done else 0.0
        return reward, done
```

Then enable HER:

```python
model.set(HER=True, batch=256, pool_size=100000)
```

The base class automatically relabels failed trajectories with achieved states as substitute goals.

## Supporting Multi-Agent RL (MARL)

For multi-agent scenarios, define a per-agent reward/done function:

```python
class MADDPG(nn.RL):
    def __init__(...):
        ...
        self.env = multi_agent_env  # Custom multi-agent env

    def reward_done_func_ma(self, rewards, dones):
        # rewards: list/array of rewards for each agent
        # dones: list/array of done flags for each agent
        return rewards, dones
```

Then enable MARL:

```python
model.set(MARL=True, batch=64)
```

The base class handles joint state/action processing and centralized training.

## Visualization & Evaluation

```python
# Plot training curves
model.visualize_reward()
model.visualize_loss()
model.visualize_reward_loss()

# Animate the trained agent
model.animate_agent(max_steps=500)

# Manual save/load
model.save('my_dqn.dat')
model.save_param('my_dqn_params.dat')

# Restore
model.restore('my_dqn.dat')
```

This pattern — inheriting from `nn.RL`, defining `action`, `__call__`, and optionally `update_param` + custom reward functions — allows you to rapidly prototype state-of-the-art agents while leveraging the full power of the framework (parallel collection, prioritized replay, distributed training, adaptive hyperparameters, etc.).

# LRFinder:
**Usage:**

Create a Note_rl agent, then execute this code:
```python
from Note_rl.lr_finder import LRFinder
# agent is a Note_rl agent
agent.optimizer = tf.keras.optimizers.Adam()
lr_finder = LRFinder(agent)

# Train a agent with 77 episodes
# with learning rate growing exponentially from 0.0001 to 1
# N: Total number of iterations (or mini-batch steps) over which the learning rate is increased.
#    This parameter determines how many updates occur between the starting learning rate (start_lr)
#    and the ending learning rate (end_lr). The learning rate is increased exponentially by a fixed
#    multiplicative factor computed as:
#         factor = (end_lr / start_lr) ** (1.0 / N)
#    This ensures that after N updates, the learning rate will reach exactly end_lr.
#
# window_size: The size of the sliding window (i.e., the number of most recent episodes)
#              used to compute the moving average and standard deviation of the rewards.
#              This normalization helps smooth out the reward signal and adjust for the fact that
#              early episodes may have lower rewards (due to limited experience) compared to later ones.
#              By using only the recent window_size rewards, we obtain a more stable and current estimate
#              of the reward statistics for normalization.
lr_finder.find(train_loss, pool_network=False, N=77, window_size=7, start_lr=0.0001, end_lr=1, episodes=77)
```
or
```python
from Note_rl.lr_finder import LRFinder
# agent is a Note_rl agent
agent.optimizer = tf.keras.optimizers.Adam()
strategy = tf.distribute.MirroredStrategy()
lr_finder = LRFinder(agent)

# Train a agent with 77 episodes
# with learning rate growing exponentially from 0.0001 to 1
# N: Total number of iterations (or mini-batch steps) over which the learning rate is increased.
#    This parameter determines how many updates occur between the starting learning rate (start_lr)
#    and the ending learning rate (end_lr). The learning rate is increased exponentially by a fixed
#    multiplicative factor computed as:
#         factor = (end_lr / start_lr) ** (1.0 / N)
#    This ensures that after N updates, the learning rate will reach exactly end_lr.
#
# window_size: The size of the sliding window (i.e., the number of most recent episodes)
#              used to compute the moving average and standard deviation of the rewards.
#              This normalization helps smooth out the reward signal and adjust for the fact that
#              early episodes may have lower rewards (due to limited experience) compared to later ones.
#              By using only the recent window_size rewards, we obtain a more stable and current estimate
#              of the reward statistics for normalization.
lr_finder.find(pool_network=False, strategy=strategy, N=77, window_size=7, start_lr=0.0001, end_lr=1, episodes=77)
```
```python
# Plot the reward, ignore 20 batches in the beginning and 5 in the end
lr_finder.plot_reward(n_skip_beginning=20, n_skip_end=5)
```
```python
# Plot rate of change of the reward
# Ignore 20 batches in the beginning and 5 in the end
# Smooth the curve using simple moving average of 20 batches
# Limit the range for y axis to (-0.02, 0.01)
lr_finder.plot_reward_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
```

# OptFinder:
**Usage:**

Create a Note agent, then execute this code:
```python
from Note_rl.opt_finder import OptFinder
# agent is a Note agent
optimizers = [tf.keras.optimizers.Adam(), tf.keras.optimizers.AdamW(), tf.keras.optimizers.Adamax()]
opt_finder = OptFinder(agent, optimizers)

# Train a agent with 7 episodes
opt_finder.find(train_loss, pool_network=False, episodes=7)
```
or
```python
from Note_rl.opt_finder import OptFinder
# agent is a Note agent
optimizers = [tf.keras.optimizers.Adam(), tf.keras.optimizers.AdamW(), tf.keras.optimizers.Adamax()]
strategy = tf.distribute.MirroredStrategy()
opt_finder = OptFinder(agent, optimizers)

# Train a agent with 7 episodes
opt_finder.find(pool_network=False, strategy=strategy, episodes=7)
```

# ParallelFinder:

**Overview**

The **AgentFinder** class is designed for reinforcement learning or multi-agent training scenarios. It trains multiple agents in parallel and selects the best performing agent based on a chosen metric (reward or loss). The class employs multiprocessing to run each agent’s training in its own process and uses callbacks at the end of each episode to update performance logs. Depending on the selected metric, at the end of the training episodes, it computes the mean reward or mean loss for each agent and updates the shared logs with the best optimizer and corresponding performance value.

---

**Key Attributes**

- **agents**  
  *Type:* `list`  
  *Description:* A list of agent instances to be trained. Each agent will run its training in a separate process.

- **optimizers**  
  *Type:* `list`  
  *Description:* A list of optimizers corresponding to the agents, used during the training process.

- **rewards**  
  *Type:* Shared dictionary (created via `multiprocessing.Manager().dict()`)  
  *Description:* Records the reward values for each episode for every agent. For each agent, a list of rewards is maintained.

- **losses**  
  *Type:* Shared dictionary  
  *Description:* Records the loss values for each episode for every agent. For each agent, a list of losses is maintained.

- **logs**  
  *Type:* Shared dictionary  
  *Description:* Stores key training information. Initially, it contains:
  - `best_reward`: Set to a very low value (-1e9) to store the best mean reward.
  - `best_loss`: Set to a high value (1e9) to store the lowest mean loss.
  - When training is complete, it also stores `best_opt`, which corresponds to the optimizer of the best performing agent.

- **lock**  
  *Type:* `multiprocessing.Lock`  
  *Description:* A multiprocessing lock used to ensure data consistency and thread safety when multiple processes update the shared dictionaries.

- **episode**  
  *Type:* `int`  
  *Description:* The total number of training episodes, set in the `find` method. This value is used to determine if the current episode is the final one.

---

**Main Methods**

**1. `__init__(self, agents, optimizers)`**

**Purpose:**  
Initializes an AgentFinder instance by setting the list of agents and corresponding optimizers. It also creates shared dictionaries for rewards, losses, and logs, and initializes a multiprocessing lock to ensure safe data access.

**Parameters:**
- `agents`: A list of agent instances.
- `optimizers`: A list of optimizers corresponding to the agents.

**Details:**  
The constructor uses `multiprocessing.Manager()` to create shared dictionaries (`rewards`, `losses`, `logs`) and sets initial values for best reward and best loss for subsequent comparisons. A lock object is created to synchronize updates in a multiprocessing environment.

**2. `on_episode_end(self, episode, logs, agent=None, lock=None)`**

**Purpose:**  
This callback function is invoked at the end of each episode when the metric is set to 'reward'. It updates the corresponding agent’s reward list and, if the episode is the last one, calculates the mean reward. If the mean reward exceeds the current best reward recorded in the shared logs, it updates the logs with the new best reward and the corresponding optimizer.

**Parameters:**
- `episode`: The current episode number (starting from 0).
- `logs`: A dictionary containing training information for the current episode; it must include the key `'reward'`.
- `agent`: The current agent instance, used to update the reward list and access its optimizer.
- `lock`: The multiprocessing lock used to synchronize access to shared data.

**Key Logic:**
1. Acquire the lock with `lock.acquire()` to ensure safe data updates.
2. Retrieve the current episode’s reward from `logs`.
3. Append the reward to the corresponding agent’s list in the `rewards` dictionary.
4. If this is the last episode (i.e., `episode + 1 == self.episode`), calculate the mean reward.
5. If the mean reward is higher than the current `best_reward` in the shared logs, update `logs['best_reward']` and `logs['best_opt']` (using the agent’s optimizer).
6. Release the lock using `lock.release()`.

**3. `on_episode_end_(self, episode, logs, agent=None, lock=None)`**

**Purpose:**  
This callback function is used when the metric is set to 'loss'. It updates the corresponding agent’s loss list and, at the end of the final episode, computes the mean loss. If the mean loss is lower than the current best loss recorded in the shared logs, it updates the logs with the new best loss and the corresponding optimizer.

**Parameters:**
- `episode`: The current episode number (starting from 0).
- `logs`: A dictionary containing training information for the current episode; it must include the key `'loss'`.
- `agent`: The current agent instance.
- `lock`: The multiprocessing lock used to synchronize access to shared data.

**Key Logic:**
1. Acquire the lock to ensure safe updates.
2. Retrieve the loss from `logs` and append it to the corresponding agent’s list in the `losses` dictionary.
3. At the last episode, calculate the mean loss and compare it to the current best loss.
4. If the mean loss is lower, update `logs['best_loss']` and `logs['best_opt']` (with the agent’s optimizer).
5. Release the lock.

**4. `find(self, train_loss=None, pool_network=True, processes=None, processes_her=None, processes_pr=None, strategy=None, episodes=1, metrics='reward', jit_compile=True)`**

**Purpose:**  
Starts the training of multiple agents using multiprocessing and utilizes callback functions to update the best agent information based on the selected metric (reward or loss).

**Parameters:**
- `train_loss`: A function or parameter for computing the training loss (optional).
- `pool_network`: Boolean flag indicating whether to use a shared network pool.
- `processes`: Number of processes to be used for training (optional).
- `processes_her`: Parameters related to HER (Hindsight Experience Replay) (optional).
- `processes_pr`: Parameters possibly related to Prioritized Experience Replay (optional).
- `strategy`: Distributed training strategy (optional). If provided, the distributed training mode is used; otherwise, standard training is performed.
- `episodes`: Total number of training episodes.
- `metrics`: The metric to be used, either `'reward'` or `'loss'`. This choice determines which callback function is used.
- `jit_compile`: Boolean flag indicating whether to enable JIT compilation to speed up training.

**Key Logic:**
1. Set the total number of episodes to `self.episodes`.
2. Iterate over each agent:
   - If the selected metric is `'reward'`:
     - Use `functools.partial` to create a `partial_callback` that binds the agent, lock, and the `on_episode_end` callback.
     - Create a callback instance using `nn.LambdaCallback`.
     - Initialize the agent’s reward list in the `rewards` dictionary.
   - If the selected metric is `'loss'`:
     - Similarly, bind the `on_episode_end_` callback.
     - Initialize the agent’s loss list in the `losses` dictionary.
3. Assign the corresponding optimizer to each agent.
4. Depending on whether a `strategy` is provided, choose the training mode:
   - If `strategy` is `None`, call the agent’s `train` method with the appropriate parameters (e.g., training loss, episodes, network pool options, process parameters, callbacks, and jit_compile settings).
   - If a `strategy` is provided, call the agent’s `distributed_training` method with similar parameters and a similar callback setup.
5. Start all training processes and wait for them to complete using `join()`.

---

**Example Usage**

Below is an example demonstrating how to use AgentFinder to train multiple agents and select the best performing agent based on either reward or loss:

```python
from Note_rl.parallel_finder import ParallelFinder

# Assume agent1 and agent2 are two initialized agent instances,
# and optimizer1 and optimizer2 are their respective optimizers.
agent1 = ...  # Initialize agent 1
agent2 = ...  # Initialize agent 2
optimizer1 = ...  # Optimizer for agent 1
optimizer2 = ...  # Optimizer for agent 2

# Create lists of agents and optimizers
agents = [agent1, agent2]
optimizers = [optimizer1, optimizer2]

# Initialize the AgentFinder instance
parallel_finder = ParallelFinder(agents, optimizers)

# Assume train_loss is defined as a function or metric for calculating training loss (if needed)
train_loss = ...

# Choose the evaluation metric: 'reward' or 'loss'
metrics_choice = 'reward'  # or 'loss'

# Execute training with 10 episodes and enable JIT compilation
parallel_finder.find(
    train_loss=train_loss,
    pool_network=True,
    processes=4,
    processes_her=2,
    processes_pr=2,
    strategy=None,  # Pass None to use standard training (not distributed)
    episodes=10,
    metrics=metrics_choice,
    jit_compile=True
)

# After training, retrieve the best record from agent_finder.logs
if metrics_choice == 'reward':
    print("Best Mean Reward:", agent_finder.logs['best_reward'])
else:
    print("Best Mean Loss:", agent_finder.logs['best_loss'])
print("Best Optimizer:", agent_finder.logs['best_opt'])
```
