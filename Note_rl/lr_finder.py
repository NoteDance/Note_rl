from matplotlib import pyplot as plt
import math
import tensorflow as tf
from Note.rl.lambda_callback import LambdaCallback
from Note.rl.assign_param import assign_param
import keras.backend as K
from tensorflow.python.util import nest
import numpy as np


class LRFinder:
    """
    Plots the change of the loss function of a Keras agent when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """

    def __init__(self, agent):
        self.agent = agent
        self.rewards = []
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9
        self.best_reward = -1e9
            
    def on_episode_end(self, episode, logs):
        if type(self.agent.optimizer)!=list:
            lr = K.get_value(self.agent.optimizer.lr)
        else:
            lr = K.get_value(self.agent.optimizer[-1].lr)
        self.lrs.append(lr)
        reward = logs['reward']
        self.rewards.append(reward)
        
        if len(self.rewards) >= self.window_size:
            recent_rewards = self.rewards[-self.window_size:]
        else:
            recent_rewards = self.rewards
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards) + 1e-8
        normalized_reward = (reward - mean_reward) / std_reward

        if episode > 5 and (math.isnan(normalized_reward) or normalized_reward < self.best_reward * 0.5):
            self.stop_training = True
            return

        if normalized_reward > self.best_reward:
            self.best_reward = normalized_reward

        lr = lr * self.factor
        if type(self.agent.optimizer)!=list:
            K.set_value(self.agent.optimizer.lr, lr)
        else:
            K.set_value(self.agent.optimizer[-1].lr, lr)
    
    def on_batch_end(self, batch, logs):
        # Log the learning rate
        if type(self.agent.optimizer)!=list:
            lr = K.get_value(self.agent.optimizer.lr)
        else:
            lr = K.get_value(self.agent.optimizer[-1].lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.agent.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        lr = lr * self.factor
        if type(self.agent.optimizer)!=list:
            K.set_value(self.agent.optimizer.lr, lr)
        else:
            K.set_value(self.agent.optimizer[-1].lr, lr)

    def find(self, train_loss=None, pool_network=True, processes=None, processes_her=None, processes_pr=None, strategy=None, N=None, window_size=None, start_lr=None, end_lr=None, episodes=1, metrics='reward', jit_compile=True):
        self.factor = (end_lr / start_lr) ** (1.0 / N)
        self.window_size = window_size
        # Save weights into a file
        initial_weights = [tf.Variable(param.read_value()) for param in nest.flatten(self.agent.param)]

        # Remember the original learning rate
        if type(self.agent.optimizer)!=list:
            original_lr = K.get_value(self.agent.optimizer.lr)
        else:
            original_lr = K.get_value(self.agent.optimizer[-1].lr)

        # Set the initial learning rate
        if type(self.agent.optimizer)!=list:
            K.set_value(self.agent.optimizer.lr, start_lr)
        else:
            K.set_value(self.agent.optimizer[-1].lr, start_lr)

        if metrics == 'reward':
            callback = LambdaCallback(on_episode_end=lambda episode, logs: self.on_episode_end(episode, logs))
        elif metrics == 'loss':
            callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        if strategy == None:
            self.agent.train(train_loss=train_loss, 
                           episodes=episodes,
                           pool_network=pool_network,
                           processes=processes,
                           processes_her=processes_her,
                           processes_pr=processes_her,
                           callbacks=[callback],
                           jit_compile=jit_compile)
        else:
            self.agent.distributed_training(strategy=strategy,
                           episodes=episodes,
                           pool_network=pool_network,
                           processes=processes,
                           processes_her=processes_her,
                           processes_pr=processes_her,
                           callbacks=[callback],
                           jit_compile=jit_compile)

        # Restore the weights to the state before agent fitting
        assign_param(self.agent.param, initial_weights)

        # Restore the original learning rate
        if type(self.agent.optimizer)!=list:
            K.set_value(self.agent.optimizer.lr, original_lr)
        else:
            K.set_value(self.agent.optimizer[-1].lr, original_lr)
    
    def plot_reward(self, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
        """
        Plots the reward.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("reward")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.rewards[n_skip_beginning:-n_skip_end])
        plt.xscale(x_scale)
        plt.show()

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale(x_scale)
        plt.show()
    
    def plot_reward_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the reward.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        plt.ylabel("rate of reward change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(lrs, derivatives)
        plt.xscale('log')
        plt.ylim(y_lim)
        plt.show()

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives_loss(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(lrs, derivatives)
        plt.xscale('log')
        plt.ylim(y_lim)
        plt.show()
    
    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.rewards[i] - self.rewards[i - sma]) / sma)
        return derivatives

    def get_derivatives_loss(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]
