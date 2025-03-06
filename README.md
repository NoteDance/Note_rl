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

# Train:
**Keras:**
Agent built with Keras.
```python
import tensorflow as tf
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.keras.DQN import DQN

model=DQN(4,128,2)
model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=False)

# If set criterion.
# model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.train(train_loss, optimizer, 100, pool_network=False)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.train(train_loss, optimizer, 100, pool_network=False)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.train(train_loss, optimizer, 100, pool_network=False)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.train(train_loss, optimizer, 100, pool_network=False)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
```python
# Use PPO.
import tensorflow as tf
from Note_rl.policy import SoftmaxPolicy
from Note_rl.examples.keras.PPO import PPO

model=PPO(4,128,2,0.7,0.7)
model.set(policy=SoftmaxPolicy(),pool_size=10000,batch=64,update_steps=1000,PPO=True)
optimizer = [tf.keras.optimizers.Adam(1e-4),tf.keras.optimizers.Adam(5e-3)]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=False)
```
```python
# Use HER.
import tensorflow as tf
from Note_rl.noise import GaussianWhiteNoiseProcess
from Note_rl.examples.keras.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(noise=GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,criterion=-5,trial_count=10,HER=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 2000, pool_network=False)
```
```python
# Use Multi-agent reinforcement learning.
import tensorflow as tf
from Note_rl.policy import SoftmaxPolicy
from Note_rl.examples.keras.MADDPG import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(policy=SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MARL=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=False)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.keras.pool_network.DQN import DQN

model=DQN(4,128,2,7)
model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,update_batches=17)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100, pool_network=True, processes=7)
```
**PyTorch:**
Agent built with PyTorch.
```python
import torch
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.pytorch.DQN import DQN

model=DQN(4,128,2)
model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
optimizer = torch.optim.Adam(model.param)
model.train(optimizer, 100)

# If set criterion.
# model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.train(optimizer, 100)

# If use prioritized replay.
# model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
# model.train(optimizer, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.train(optimizer, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.train(optimizer, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.train(optimizer, 100)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
```python
# Use HER.
import torch
from Note_rl.noise import GaussianWhiteNoiseProcess
from Note_rl.examples.pytorch.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(noise=GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,criterion=-5,trial_count=10,HER=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 2000)
```
```python
# Use Multi-agent reinforcement learning.
import torch
from Note_rl.policy import SoftmaxPolicy
from Note_rl.examples.pytorch.MADDPG import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set(policy=SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MARL=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 100)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import torch
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.pytorch.pool_network.DQN import DQN

model=DQN(4,128,2,7)
model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
optimizer = torch.optim.Adam(model.param)
model.train(optimizer, 100, pool_network=True, processes=7)
```
```python
# Use HER.
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import torch
from Note_rl.noise import GaussianWhiteNoiseProcess
from Note_rl.examples.pytorch.pool_network.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005,7)
model.set(noise=GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,trial_count=10,HER=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(train_loss, optimizer, 2000, pool_network=True, processes=7, processes_her=4)
```

# Distributed training:
**MirroredStrategy:**
Agent built with Keras.
```python
import tensorflow as tf
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.keras.DQN import DQN

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DQN(4,128,2)
  optimizer = tf.keras.optimizers.Adam()
model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, strategy, 100, pool_network=False)

# If set criterion.
# model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=GLOBAL_BATCH_SIZE,update_steps=10,trial_count=10,criterion=200)
# model.distributed_training(optimizer, strategy, 100, pool_network=False)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.distributed_training(optimizer, strategy, 100, pool_network=False)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.distributed_training(optimizer, strategy, 100, pool_network=False)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.distributed_training(optimizer, strategy, 100, pool_network=False)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```
```python
# Use PPO
import tensorflow as tf
from Note_rl.policy import SoftmaxPolicy
from Note_rl.examples.keras.PPO import PPO

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=PPO(4,128,2,0.7,0.7)
  optimizer = [tf.keras.optimizers.Adam(1e-4),tf.keras.optimizers.Adam(5e-3)]

model.set(policy=SoftmaxPolicy(),pool_size=10000,batch=GLOBAL_BATCH_SIZE,update_steps=1000,PPO=True)
model.distributed_training(optimizer, strategy, 100, pool_network=False)
```
```python
# Use HER.
import tensorflow as tf
from Note_rl.noise import GaussianWhiteNoiseProcess
from Note_rl.examples.keras.DDPG_HER import DDPG

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]

model.set(noise=GaussianWhiteNoiseProcess(),pool_size=10000,batch=GLOBAL_BATCH_SIZE,criterion=-5,trial_count=10,HER=True)
model.distributed_training(optimizer, strategy, 2000, pool_network=False)
```
```python
# Use Multi-agent reinforcement learning.
import tensorflow as tf
from Note_rl.policy import SoftmaxPolicy
from Note_rl.examples.keras.MADDPG import DDPG

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]

model.set(policy=SoftmaxPolicy(),pool_size=3000,batch=GLOBAL_BATCH_SIZE,trial_count=10,MARL=True)
model.distributed_training(optimizer, strategy, 100, pool_network=False)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import tensorflow as tf
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.keras.pool_network.DQN import DQN

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DQN(4,128,2,7)
  optimizer = tf.keras.optimizers.Adam()
model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=GLOBAL_BATCH_SIZE,update_batches=17)
model.distributed_training(optimizer, strategy, 100, pool_network=True, processes=7)
```
**MultiWorkerMirroredStrategy:**
```python
import tensorflow as tf
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.keras.pool_network.DQN import DQN
import sys
import os

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

strategy = tf.distribute.MultiWorkerMirroredStrategy()
per_worker_batch_size = 64
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers

with strategy.scope():
  multi_worker_model = DQN(4,128,2)
  optimizer = tf.keras.optimizers.Adam()

multi_worker_model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=global_batch_size,update_batches=17)
multi_worker_model.distributed_training(optimizer, strategy, num_episodes=100,
                    pool_network=True, processes=7)

# If set criterion.
# model.set(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=global_batch_size,update_steps=10,trial_count=10,criterion=200)
# multi_worker_model.distributed_training(optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# multi_worker_model.distributed_training(optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# multi_worker_model.distributed_training(optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# multi_worker_model.distributed_training(optimizer, strategy, num_episodes=100,
#                    pool_network=True, processes=7)

# visualize
# model.visualize_loss()
# model.visualize_reward()
# model.visualize_reward_loss()

# animate agent
# model.animate_agent(200)

# save
# model.save_param('param.dat')
# model.save('model.dat')
```

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
from Note import nn
# agent is a Note agent
optimizers = [tf.keras.optimizers.Adam(), tf.keras.optimizers.AdamW(), tf.keras.optimizers.Adamax()]
opt_finder = nn.OptFinder(agent, optimizers)

# Train a agent with 7 episodes
opt_finder.find(train_loss, pool_network=False, episodes=7)
```
or
```python
from Note import nn
# agent is a Note agent
optimizers = [tf.keras.optimizers.Adam(), tf.keras.optimizers.AdamW(), tf.keras.optimizers.Adamax()]
strategy = tf.distribute.MirroredStrategy()
opt_finder = nn.OptFinder(agent, optimizers)

# Train a agent with 7 episodes
opt_finder.find(pool_network=False, strategy=strategy, episodes=7)
```
