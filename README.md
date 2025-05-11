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

# Save model parameters:
```python
import pickle
output_file=open('param.dat','wb')
pickle.dump(model.param,output_file)
output_file.close()
```
or
```python
model = MyModel(...)
model.save_param('param.dat')
```

# Restore model parameters:
```python
import pickle
input_file=open('param.dat','rb')
param=pickle.load(input_file)
input_file.close()
```
or
```python
model = MyModel(...)
model.restore_param('param.dat')
```
or
```python
from Note import nn
param=nn.restore_param('param.dat')
```

# Save model:
```python
model = MyModel(...)
model.save('model.dat')
```

# Restore model:
```python
# distributed training
with strategy.scope():
    model = MyModel(...)
    model.restore('model.dat')
```
or
```python
model = MyModel(...)
model.restore('model.dat')
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
