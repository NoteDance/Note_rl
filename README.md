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

# RL.train:

**Description**:
Runs the main training loop for the `RL` agent. Supports single-process and multi-process experience collection via a **pool network**, distributed training strategies (Mirrored/MultiWorker/ParameterServer), just-in-time compilation for training steps, callbacks, and special replay mechanisms: Hindsight Experience Replay (HER), Prioritized Replay (PR) and PPO-compatible behavior. The method coordinates environment rollout(s), buffer aggregation, batch sampling, training updates, optional periodic trimming of replay buffers (via `window_size_fn` / `window_size_ppo`), logging and model saving.

**Arguments**:

* **`train_loss`** (`tf.keras.metrics.Metric`): Metric used to accumulate/report training loss (e.g. `tf.keras.metrics.Mean()`).
* **`optimizer`** (`tf.keras.optimizers.Optimizer` or list): Optimizer (or list of optimizers) used to apply gradients. If `self.optimizer` is already set, the passed `optimizer` is only used to initialize `self.optimizer` (see code behaviour).
* **`episodes`** (`int`, optional): Number of episodes to run. If `None`, training runs indefinitely (or until `self.stop_training` or reward criterion is met).
* **`jit_compile`** (`bool`, optional, default=`True`): Whether to use `@tf.function(jit_compile=True)` compiled train steps. When True the compiled train-steps are used where available.
* **`pool_network`** (`bool`, optional, default=`True`): Enable pool-network multi-process rollouts. When True, experiences are collected in parallel by `processes` worker processes and aggregated into shared (manager) buffers.
* **`processes`** (`int`, optional): Number of parallel worker processes used when `pool_network=True` to collect experience.
* **`processes_her`** (`int`, optional): When HER is enabled, number of processes used for HER batch generation. Affects internal multiprocessing logic and intermediate buffers.
* **`processes_pr`** (`int`, optional): When PR is enabled, number of processes used for prioritized replay sampling. Affects internal multiprocessing logic and intermediate buffers.
* **`window_size`** (`int`, optional): Fixed window size used when trimming per-process buffers inside `pool` / `store_in_parallel`. (If `None` uses default popping behavior.)
* **`clearing_freq`** (`int`, optional): When set, triggers periodic trimming of per-process buffers every `clearing_freq` stored items.
* **`window_size_`** (`int`, optional): A global fallback window size used in several trimming spots when buffers exceed `self.pool_size`.
* **`window_size_ppo`** (`int`, optional): Default PPO-specific window trimming size used if `window_size_fn` is not supplied (used when `PPO == True` and `PR == True`).
* **`random`** (`bool`, optional, default=`False`): When `pool_network=True`, toggles random worker selection vs. inverse-length selection logic used in `store_in_parallel`.
* **`save_data`** (`bool`, optional, default=`True`): If True, keeps collected pool lists in shared manager lists to allow saving/resuming; otherwise per-process buffers are reinitialized each run.
* **`p`** (`int`, optional): Controls the logging/printing frequency. If `p` is `None` a default of 9 is used (internally the implementation derives a logging interval). If `p == 0` the periodic logging block is disabled (the code contains `if p!=0` guards around prints).
  *Implementation note:* The code transforms the user-supplied `p` into an internal `self.p` and a derived integer `p` that is used for printing interval computation (`p` becomes roughly the number of episodes between logs).

**Returns**:

* If running with `distributed_flag==True`: returns `(total_loss / num_batches).numpy()` (the average distributed loss for the epoch/batch group).
* Otherwise: returns `train_loss.result().numpy()` (the metric's current value).
* If early exit happens (e.g. `self.stop_training==True`), the function returns early (commonly the current `train_loss` value or `np.array(0.)` depending on branch).

**Details**:

1. **Initialization & manager setup**:

   * If `pool_network=True`, a `multiprocessing.Manager()` is created and many local lists/buffers (`state_pool_list`, `action_pool_list`, `reward_pool_list`, etc.) are converted into manager lists/dicts so worker processes can append data safely.
   * Per-process data structures (e.g. `self.ratio_list`, `self.TD_list`) are initialized if `PR==True`. When `PPO==True` and `PR==True` the code uses per-process `ratio_list` / `TD_list` and later concatenates them into `self.prioritized_replay` before training.

2. **Callbacks & training lifecycle**:

   * Calls `on_train_begin` on registered callbacks at the start.
   * Per-episode: calls `on_episode_begin` and `on_episode_end` callbacks with logs including `'loss'` and `'reward'`.
   * Per-batch: calls `on_batch_begin` / `on_batch_end` with batch logs (loss). This applies to both the PR/HER per-batch generation branches and the dataset-driven branches.
   * Respects `self.stop_training` — if set True during training the method exits early and returns.

3. **Experience collection**:

   * When `pool_network=True` the function spawns `processes` worker processes (each runs `store_in_parallel`) to produce per-process pool lists, then `concatenate`s them (or packs them into `self.state_pool[7]` etc. when `processes_pr`/`processes_her` are used).
   * If `processes_pr`/`processes_her` are set, special per-process lists (`self.state_list`, `self.action_list`, ...) are used for parallel sampling and later aggregated in `data_func()`.

4. **Training procedure & batching**:

   * Two main modes:

     * **PR/HER path**: When `self.PR` or `self.HER` is `True`, batches are generated via `self.data_func()` (which may itself spawn worker processes to form batches). The loop iterates over `batches` computed from the pool length / `self.batch`. Each generated batch is turned into a small `tf.data.Dataset` (batched to `self.global_batch_size`) and then:

       * If using a MirroredStrategy, the dataset is distributed and `distributed_train_step` or `_` is used.
       * Else the code uses `train_step` / `train_step_` or directly the non-distributed loops.
     * **Plain dataset path**: When not PR/HER, the code creates a `tf.data.Dataset` from the entire pool (`self.state_pool,...`) and iterates it as usual (shuffle when not `pool_network`), applying `train_step`/`train_step_` for each mini-batch.
   * `self.batch_counter` and `self.step_counter` are used to decide when to call `self.update_param()` and (if PPO + PR) when to apply `window_size_fn` / `window_size_ppo` trimming to per-process buffers.

5. **Distributed strategies**:

   * Code supports `tf.distribute.MirroredStrategy`, `MultiWorkerMirroredStrategy` and `ParameterServerStrategy` integration:

     * When MirroredStrategy is detected, datasets are distributed via `strategy.experimental_distribute_dataset` and `distributed_train_step` is used.
     * For `MultiWorkerMirroredStrategy` a custom path calls `self.CTL` (user-defined) to compute loss over multiple workers.
     * If a ParameterServerStrategy is used and `stop_training` triggers, the code may call `self.coordinator.join()` to sync workers and exit.

6. **Priority replay (PR) & PPO interactions**:

   * If `PR==True` and `PPO==True`, the training loop:

     * Maintains per-process `ratio_list` / `TD_list` during collection.
     * Concatenates them into `self.prioritized_replay.ratio` and `self.prioritized_replay.TD` before sampling/training.
     * When `self.batch_counter % self.update_batches == 0` or `self.update_steps` triggers an update, the code attempts to call `self.window_size_fn(p)` (if provided) for each process and trims per-process buffers to the returned `window_size` (or uses `window_size_ppo` fallback). This enables adaptive trimming (e.g. driven by ESS).
   * If `PR==True` but `PPO==False`, only `TD_list` is used/concatenated.

7. **Saving & early stopping**:

   * Periodic saving: if `self.path` is set and `i % self.save_freq == 0`, calls `save_param_` or `save_` depending on `self.save_param_only`. `max_save_files` and `save_best_only` can be used in your saving implementations (not implemented here).
   * Reward-based termination: if `self.trial_count` and `self.criterion` are set, the method computes `avg_reward` over the most recent `trial_count` episodes and will terminate early when `avg_reward >= criterion`. It prints summary info (episode count, average reward, elapsed time) and returns.

8. **Logging behavior**:

   * The printed logs (loss/reward) are gated by the derived `p` logic. Passing `p==0` suppresses periodic printouts (there are many `if p!=0` guards around prints).
   * The method always updates `self.loss_list`, `self.total_episode`, and `self.time` counters.

9. **Return values & possible early-exit values**:

   * On normal epoch/episode completion the method returns the computed train loss (distributed average or `train_loss.result().numpy()`).
   * On early exit (stop\_training true or ParameterServer coordinator join) the method may return `np.array(0.)` or the current metric depending on branch.

**Notes / Implementation caveats**:

* The `p` parameter behavior is non-standard: if you want the default printing cadence, pass `p=None` (internally becomes 9). Pass `p=0` to disable periodic printing.
* When `PR==True` and `PPO==True` the code expects per-process `ratio_list`/`TD_list` and relies on concatenation. Make sure those variables are initialized and that `self.window_size_fn` (if used) handles small buffer sizes (the user-provided `window_size_fn` should guard `len(weights) < 2`).
* Be defensive around buffer sizes: many places assume `len(self.state_pool) >= self.batch`. During warm-up training you may see early returns if the pool is not yet filled.
* The method mutates internal buffers when trimming; ensure that any external references to those buffers are updated if needed (they are manager lists/dicts in `pool_network` mode).
* Callbacks are integrated; use them for logging, checkpointing, early stopping, or custom monitoring.

# RL.distributed\_training

**Description**
Runs a distributed / multi-device training loop for the `RL` agent using TensorFlow `tf.distribute` strategies. It combines multi-process environment rollouts (pool network) with distributed model updates (MirroredStrategy / MultiWorkerMirroredStrategy) and supports special replay modes (Prioritized Replay `PR`, Hindsight ER `HER`) and PPO interactions. The method orchestrates rollout collection across OS processes, constructs aggregated replay buffers, builds distributed datasets, runs distributed train steps, calls callbacks, does periodic trimming (via `window_size_fn` / `window_size_ppo`), saving, and early stopping.

---

## Arguments

* **`optimizer`** (`tf.keras.optimizers.Optimizer` or list): Optimizer(s) to apply gradients. If `self.optimizer` is `None` this will initialize `self.optimizer`.
* **`strategy`** (`tf.distribute.Strategy`): A TensorFlow distribution strategy instance (e.g. `tf.distribute.MirroredStrategy`, `tf.distribute.MultiWorkerMirroredStrategy`) under whose scope distributed training is executed.
* **`episodes`** (`int`, optional): Number of episodes to run (MirroredStrategy path). If `None` and `num_episodes` supplied, `num_episodes` may be used by some branches.
* **`num_episodes`** (`int`, optional): Alternative name for `episodes` used by some strategy branches (e.g. MultiWorker path). If provided, it overrides/assigns `episodes`.
* **`jit_compile`** (`bool`, optional, default=`True`): Whether to use JIT compiled train steps where available (`@tf.function(jit_compile=True)`).
* **`pool_network`** (`bool`, optional, default=`True`): Enable multi-process environment rollouts (pool of worker processes).
* **`processes`** (`int`, optional): Number of parallel worker processes to launch for rollouts when `pool_network=True`.
* **`processes_her`** (`int`, optional): Number of worker processes dedicated for HER sampling (if `HER=True`).
* **`processes_pr`** (`int`, optional): Number of worker processes dedicated for PR sampling (if `PR=True`).
* **`window_size`** (`int`, optional): Fixed per-process trimming window used in collection logic.
* **`clearing_freq`** (`int`, optional): Periodic trimming frequency (applies to per-process buffers).
* **`window_size_`** (`int`, optional): Global fallback window used in some trimming branches.
* **`window_size_ppo`** (`int`, optional): Default PPO window trimming fallback used if `window_size_fn` is not present (used with `PPO==True and PR==True`).
* **`random`** (`bool`, optional, default=`False`): Controls per-process selection strategy in `store_in_parallel` (random vs. inverse-length selection).
* **`save_data`** (`bool`, optional, default=`True`): Whether to persist per-process buffers to a `multiprocessing.Manager()` so they survive across processes and can be saved.
* **`p`** (`int`, optional): Controls printing/logging frequency. If `None` an internal default is used (≈9). Passing `p==0` disables periodic printing. Internally the method transforms `p` to an interval used for logging.

---

## Returns

* For MirroredStrategy / distributed branches: returns `(total_loss / num_batches).numpy()` when `distributed_flag==True` and that branch computes `total_loss / num_batches`.
* Otherwise returns `train_loss.result().numpy()` (current metric value).
* The function may return early (e.g. `self.stop_training==True` or when reward `criterion` is met). In early-exit cases the return value depends on the branch (commonly the current metric or `np.array(0.)`).

---

## Behaviour / Details

1. **Distributed setup**

   * The function sets `self.distributed_flag = True` and defines a `compute_loss` closure inside `strategy.scope()` that calls `tf.nn.compute_average_loss` with `global_batch_size=self.batch`. This is used by the distributed train step to scale per-example losses.
   * It supports at least two strategy types explicitly:

     * `tf.distribute.MirroredStrategy` — typical synchronous multi-GPU single-machine use; the function builds distributed datasets and uses `distributed_train_step`.
     * `tf.distribute.MultiWorkerMirroredStrategy` — multi-worker synchronous training. The code follows a slightly different loop (uses `self.CTL` for loss aggregation in some branches).

2. **Pool-network (multi-process rollouts)**

   * If `pool_network=True` the method creates a `multiprocessing.Manager()` and converts `self.env` and many per-process lists into manager lists/dicts so worker processes can fill them concurrently.
   * For `PR==True` and `PPO==True` it initializes per-process `ratio_list` and `TD_list` (as `tf.Variable` wrappers) and later concatenates them into `self.prioritized_replay.ratio` / `.TD` before training.
   * Worker processes are launched using `mp.Process(target=self.store_in_parallel, args=(p, lock_list))` to collect rollouts. **Note:** the code references `lock_list` when launching workers in some branches but `lock_list` is not created in every branch of this function (this is an implementation caveat — see *Caveats*).

3. **Data aggregation & sampling**

   * When `processes_her` / `processes_pr` are provided, the code collects per-process mini-batches (`self.state_list`, `self.action_list`, etc.) and `data_func()` uses those to form training batches.
   * When not using PR/HER, per-process pools are concatenated `np.concatenate(self.state_pool_list)` etc. to form the full `self.state_pool` which is turned into a `tf.data.Dataset`.

4. **Training step selection**

   * For Mirrored strategy: dataset is wrapped with `strategy.experimental_distribute_dataset()` and the loop calls `distributed_train_step` (JIT or non-JIT variant depending on `jit_compile`).
   * For MultiWorker strategy: the code takes a different path and (in places) calls `self.CTL(multi_worker_dataset)` — a custom user-defined procedure expected to exist on the RL instance.
   * For non-distributed branches fallback to `train1` / `train2` logic is reused.

5. **PR / PPO interactions**

   * If `PR` is enabled, per-process TD / ratio lists are concatenated into the prioritized replay object before sampling/training.
   * If `PPO` + `PR` the method uses `window_size_fn` (if present) to compute adaptive trimming for each process and trims `state_pool_list[p]` etc. accordingly after update steps; otherwise it falls back to `window_size_ppo`.

6. **Callbacks, saving, and early stopping**

   * Calls callbacks: `on_train_begin`, `on_episode_begin`, `on_batch_begin`, `on_batch_end`, `on_episode_end`, `on_train_end` at appropriate points.
   * Saves model / params periodically when `self.path` is set according to `self.save_freq`.
   * If `self.trial_count` and `self.criterion` are set, computes a rolling average reward over recent episodes and stops training early if criterion is reached.

---

## Complexity & performance notes

* Launching many OS processes for rollouts can be CPU- and memory- intensive. Use a sensible `processes` count per machine.
* MirroredStrategy moves gradient application to devices — ensure your batch sizing and `global_batch_size` match your device count to avoid under/over-scaling.
* `PR` requires additional memory for the `ratio`/`TD` arrays; be mindful when concatenating per-process lists.

---

## Caveats / Implementation notes (important)

* **`lock_list` usage:** the function passes `lock_list` into `store_in_parallel` in several places but `lock_list` is not defined inside `distributed_training` before use. If you rely on locks to guard manager lists, make sure to construct `lock_list = [mp.Lock() for _ in range(processes)]` (as is done in the non-distributed `train` function) and pass it into the worker processes.
* **Small buffer sizes:** many trimming and `window_size_fn` usages assume `len(weights) >= 2`. Guard `window_size_fn` and trimming calls against tiny buffers during warm-up.
* **`self.CTL` and other user hooks:** The code calls `self.CTL(...)` in MultiWorker branches — ensure you implement this helper to compute the loss when using MultiWorker strategy.
* **Return values vary by branch:** different strategy branches return different items (distributed average loss or metric result). Tests should validate the return path you use.

# RL.adjust_batch_size:

**Description**:
This method dynamically adjusts the batch size for training based on the Effective Sample Size (ESS) of the prioritized replay buffer, which measures the diversity of sampled experiences. It uses an Exponential Moving Average (EMA) of ESS to ensure smooth adjustments. Optionally, it also adapts related hyperparameters like the priority exponent alpha, learning rate, exploration epsilon, update frequency, soft update tau, discount factor gamma, store count, weight decay, beta1, beta2, and PPO clip range, using ESS feedback to balance exploration, stability, and efficiency in reinforcement learning algorithms such as DQN or PPO.

**Arguments**:

- **`smooth`** (`float`, default=`0.2`): The smoothing coefficient for the EMA of ESS, controlling adaptation speed to new ESS values.
  
- **`batch_params`** (`dict`, optional): Dictionary for batch adjustment. Keys: `'scale'` (scaling factor, default 1.0), `'min'` (min batch, optional), `'max'` (max batch, optional), `'align'` (alignment granularity, optional).
  
- **`target_ess`** (`float`, optional): The target ESS value for adaptive computation. If provided, batch size scales with the ratio of EMA ESS to target ESS.
  
- **`alpha_params`** (`dict`, optional): Dictionary for adjusting PER priority exponent alpha. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (smoothing, default 0.2).
  
- **`lr_params`** (`dict`, optional): Dictionary for adjusting learning rates. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (smoothing, default 0.2).
  
- **`eps_params`** (`dict`, optional): Dictionary for adjusting exploration epsilon. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (smoothing, default 0.2).
  
- **`freq_params`** (`dict`, optional): Dictionary for adjusting update frequency. Keys: `'scale'` (scale), `'min'`/`'max'` (bounds).
  
- **`tau_params`** (`dict`, optional): Dictionary for adjusting soft update tau. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (smoothing, default 0.2).
  
- **`gamma_params`** (`dict`, optional): Dictionary for adjusting discount factor gamma. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (smoothing, default 0.2).
  
- **`store_params`** (`dict`, optional): Dictionary for adjusting store count. Keys: `'scale'` (scale), `'min'`/`'max'` (bounds).
  
- **`weight_decay_params`** (`dict`, optional): Dictionary for adjusting weight decay. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (smoothing, default 0.2).
  
- **`beta1_params`** (`dict`, optional): Dictionary for adjusting Adam beta1. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (smoothing, default 0.2).
  
- **`beta2_params`** (`dict`, optional): Dictionary for adjusting Adam beta2. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (smoothing, default 0.2).
  
- **`clip_params`** (`dict`, optional): Dictionary for adjusting PPO clip range. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (smoothing, default 0.2).

**Returns**:
- No return value. Updates `self.batch` and optional hyperparameters in-place.

**Details**:
1. **ESS Computation and Smoothing**:
   - Computes weights from TD errors (or PPO ratios) raised to alpha.
   - Calculates ESS as `1 / sum(p^2)` where `p = weights / sum(weights)`.
   - Applies EMA smoothing to ESS for stability.

2. **Batch Size Adjustment**:
   - If `target_ess` provided, new batch = `current * (ema / target_ess) * scale`.
   - Clips to min/max and aligns to multiples of `align`.

3. **Hyperparameter Adaptations**:
   - Calls dedicated methods for alpha, LR, epsilon, etc., using ESS ratio as feedback (high ESS → aggressive adjustments like larger LR/clip).
   - Supports multi-optimizer/policy lists; updates in-place (e.g., `self.clip = ...`).

4. **Integration Notes**:
   - Assumes `self.prioritized_replay` for PER/PPO and `self.batch` for current value.
   - Call periodically (e.g., every 10-50 steps) in training loop.

# RL.adabatch:

**Description**:
This method dynamically adjusts the batch size based on estimated gradient noise to maintain a target noise level for balanced optimization. It computes gradient variance via repeated backpropagations on a fixed batch, applies EMA smoothing, and scales batch inversely with noise (high noise → larger batch). Optionally adapts hyperparameters like alpha, LR, epsilon, update frequency, tau, gamma, weight decay, beta1, beta2, and PPO clip using noise feedback, suitable for noisy RL environments.

**Arguments**:

- **`num_samples`** (`int`, required): Number of repeated gradient computations for variance estimation.
  
- **`target_noise`** (`float`, default=`1e-3`): Target gradient noise level; adjusts batch to achieve this.
  
- **`smooth`** (`float`, default=`0.2`): EMA smoothing for noise estimate.
  
- **`batch_params`** (`dict`, optional): For batch adjustment. Keys: `'scale'` (default 1.0), `'min'`/`'max'` (bounds), `'align'` (granularity).
  
- **`alpha_params`** (`dict`, optional): For PER alpha. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (default 0.2).
  
- **`lr_params`** (`dict`, optional): For LR. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (default 0.2).
  
- **`eps_params`** (`dict`, optional): For epsilon. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (default 0.2).
  
- **`freq_params`** (`dict`, optional): For update frequency. Keys: `'scale'` (scale), `'min'`/`'max'` (bounds).
  
- **`tau_params`** (`dict`, optional): For tau. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (default 0.2).
  
- **`gamma_params`** (`dict`, optional): For gamma. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (default 0.2).
  
- **`weight_decay_params`** (`dict`, optional): For weight decay. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (default 0.2).
  
- **`beta1_params`** (`dict`, optional): For beta1. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (default 0.2).
  
- **`beta2_params`** (`dict`, optional): For beta2. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (default 0.2).
  
- **`clip_params`** (`dict`, optional): For PPO clip. Keys: `'rate'` (rate), `'min'`/`'max'` (bounds), `'smooth'` (default 0.2).
  
- **`jit_compile`** (`bool`, default=`True`): Enables JIT for gradient computation.

**Returns**:
- No return value. Updates `self.batch` and hyperparameters in-place.

**Details**:
1. **Noise Estimation**:
   - Samples fixed batch, computes gradients `num_samples` times (JIT optional).
   - Variance as noise proxy, EMA smoothed.

2. **Batch Adjustment**:
   - New batch = `current * (ema_noise / target_noise) * scale`, clipped/aligned.

3. **Hyperparameter Adaptations**:
   - Calls methods with GNS=True for conservative adjustments (high noise → smaller LR/clip, larger decay/beta).

4. **Integration Notes**:
   - Supports HER/PR buffers (index 7); call after buffer fill.

# RL.adjust:

**Description**:
This wrapper method unifies ESS-based and GNS-based adjustments by dispatching to `adjust_batch_size` or `adabatch` based on input. If `target_noise` is provided, it uses GNS for noise-driven adaptations; otherwise, ESS for diversity-driven ones. It enables flexible hyperparameter tuning in RL training loops.

**Arguments**:

- **`target_ess`** (`float`, optional): Target ESS for diversity-based adjustments.
  
- **`target_noise`** (`float`, optional): Target noise for variance-based adjustments; triggers GNS mode.
  
- **`num_samples`** (`int`, optional): For GNS estimation (required if `target_noise` provided).
  
- **`smooth`** (`float`, default=`0.2`): EMA smoothing for ESS or noise.
  
- **`batch_params`** (`dict`, optional): For batch scaling/bounds/alignment.
  
- **`alpha_params`** (`dict`, optional): For PER alpha.
  
- **`lr_params`** (`dict`, optional): For learning rate.
  
- **`eps_params`** (`dict`, optional): For epsilon.
  
- **`freq_params`** (`dict`, optional): For update frequency.
  
- **`tau_params`** (`dict`, optional): For soft update tau.
  
- **`gamma_params`** (`dict`, optional): For discount gamma.
  
- **`store_params`** (`dict`, optional): For store count.
  
- **`weight_decay_params`** (`dict`, optional): For weight decay.
  
- **`beta1_params`** (`dict`, optional): For Adam beta1.
  
- **`beta2_params`** (`dict`, optional): For Adam beta2.
  
- **`clip_params`** (`dict`, optional): For PPO clip.
  
- **`jit_compile`** (`bool`, default=`True`): For GNS gradient computation.

**Returns**:
- No return value. Dispatches to underlying methods for in-place updates.

**Details**:
1. **Mode Selection**:
   - If `target_noise` provided, calls `adabatch` for GNS-based adjustments.
   - Else, calls `adjust_batch_size` for ESS-based.

2. **Unified Feedback**:
   - Passes params to sub-methods; GNS mode uses noise variance as proxy.

3. **Hyperparameter Handling**:
   - Supports multi-optimizer/policy; updates in-place.

4. **Integration Notes**:
   - Call in training loop (e.g., every 50 steps); assumes filled buffer.

**Usage Example**:

https://github.com/NoteDance/Note_rl/blob/main/Note_rl/examples/keras/pool_network/PPO_pr.py
https://github.com/NoteDance/Note_rl/blob/main/Note_rl/examples/pytorch/pool_network/PPO_pr.py

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
