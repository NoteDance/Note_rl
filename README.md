# Introduction:
This libaray allows you to easily train agents built with Keras or PyTorch using reinforcement learning.You just need to have your agent class inherit from the RL or RL_pytorch class, and you can easily train your agent built with Keras or PyTorch. You can learn how to build an agent from the examples [here](https://github.com/NoteDance/Reinforcement-Learning/tree/main/Note_rl/examples). The README shows how to train, save, and restore agent built with Keras or PyTorch.

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
## Keras:
Agent built with Keras.
```python
import tensorflow as tf
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.keras.DQN import DQN

model=DQN(4,128,2)
model.set_up(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)

# If set criterion.
# model.set_up(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.train(train_loss, optimizer, 100)

# If use prioritized replay.
# model.set_up(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
# model.train(train_loss, optimizer, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.train(train_loss, optimizer, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.train(train_loss, optimizer, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.train(train_loss, optimizer, 100)

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

model=PPO(4,128,2,0.7,0.7)
model.set_up(policy=SoftmaxPolicy(),pool_size=10000,batch=64,update_steps=1000,PPO=True)
optimizer = [tf.keras.optimizers.Adam(1e-4),tf.keras.optimizers.Adam(5e-3)]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)
```
```python
# Use HER
import tensorflow as tf
from Note_rl.noise import GaussianWhiteNoiseProcess
from Note_rl.examples.keras.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set_up(noise=GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,criterion=-5,trial_count=10,HER=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 2000)
```
```python
# Use Multi-agent reinforcement learning
import tensorflow as tf
from Note_rl.policy import SoftmaxPolicy
from Note_rl.examples.keras.MADDPG import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set_up(policy=SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MA=True)
optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]
train_loss = tf.keras.metrics.Mean(name='train_loss')
model.train(train_loss, optimizer, 100)
```
## PyTorch:
Agent built with PyTorch.
```python
import torch
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.pytorch.DQN import DQN

model=DQN(4,128,2)
model.set_up(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
optimizer = torch.optim.Adam(model.param)
model.train(optimizer, 100)

# If set criterion.
# model.set_up(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.train(optimizer, 100)

# If use prioritized replay.
# model.set_up(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
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
# Use HER
import torch
from Note_rl.noise import GaussianWhiteNoiseProcess
from Note_rl.examples.pytorch.DDPG_HER import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set_up(noise=GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,criterion=-5,trial_count=10,HER=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 2000)
```
```python
# Use Multi-agent reinforcement learning
import torch
from Note_rl.policy import SoftmaxPolicy
from Note_rl.examples.pytorch.MADDPG import DDPG

model=DDPG(128,0.1,0.98,0.005)
model.set_up(policy=SoftmaxPolicy(),pool_size=3000,batch=32,trial_count=10,MA=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
model.train(optimizer, 100)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
import torch
from Note_rl.policy import EpsGreedyQPolicy
from Note_rl.examples.pytorch.multiprocessing.DQN import DQN
import multiprocessing as mp

model=DQN(4,128,2,7)
model.set_up(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_batches=17)
optimizer = torch.optim.Adam(model.param)
manager=mp.Manager()
model.train(optimizer, 100, mp=mp, manager=manager, processes=7)
```
```python
# This technology uses Python’s multiprocessing module to speed up trajectory collection and storage, I call it Pool Network.
# Furthermore use Python’s multiprocessing module to speed up getting a batch of data.
import torch
from Note_rl.noise import GaussianWhiteNoiseProcess
from Note_rl.examples.pytorch.multiprocessing.DDPG_HER import DDPG
import multiprocessing as mp

model=DDPG(128,0.1,0.98,0.005,7)
model.set_up(noise=GaussianWhiteNoiseProcess(),pool_size=10000,batch=256,trial_count=10,HER=True)
optimizer = [torch.optim.Adam(model.param[0]),torch.optim.Adam(model.param[1])]
manager=mp.Manager()
model.train(train_loss, optimizer, 2000, mp=mp, manager=manager, processes=7, processes_her=4)
```

# Distributed training:
## Keras:
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
model.set_up(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

# If set criterion.
# model.set_up(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200)
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

# If use prioritized replay.
# model.set_up(policy=EpsGreedyQPolicy(0.01),pool_size=10000,batch=64,update_steps=10,trial_count=10,criterion=200,PR=True,initial_TD=7,alpha=0.7)
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

# If save the model at intervals of 10 episode, with a maximum of 2 saved file, and the file name is model.dat.
# model.path='model.dat'
# model.save_freq=10
# model. max_save_files=2
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

# If save parameters only
# model.path='param.dat'
# model.save_freq=10
# model. max_save_files=2
# model.save_param_only=True
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

# If save best only
# model.path='model.dat'
# model.save_best_only=True
# model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)

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

model.set_up(policy=SoftmaxPolicy(),pool_size=10000,update_steps=1000,PPO=True)
model.distributed_training(GLOBAL_BATCH_SIZE, optimizer, 100)
```
```python
# Use HER
import tensorflow as tf
from Note_rl.noise import GaussianWhiteNoiseProcess
from Note_rl.examples.keras.DDPG_HER import DDPG

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]

model.set_up(noise=GaussianWhiteNoiseProcess(),pool_size=10000,criterion=-5,trial_count=10,HER=True)
model.train(GLOBAL_BATCH_SIZE, optimizer, 2000)
```
```python
# Use Multi-agent reinforcement learning
import tensorflow as tf
from Note_rl.policy import SoftmaxPolicy
from Note_rl.examples.keras.MADDPG import DDPG

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

with strategy.scope():
  model=DDPG(128,0.1,0.98,0.005)
  optimizer = [tf.keras.optimizers.Adam(),tf.keras.optimizers.Adam()]

model.set_up(policy=SoftmaxPolicy(),pool_size=3000,trial_count=10,MA=True)
model.train(GLOBAL_BATCH_SIZE, optimizer, 100)
```
