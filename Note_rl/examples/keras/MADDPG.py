import tensorflow as tf
from Note_rl.RL import RL
from Note_rl.assign_param import assign_param
from keras.models import Sequential
from keras import Model
import numpy as np
import gym
from gym import spaces


class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents=2, state_size=4, action_size=2, done_threshold=10):
        super(MultiAgentEnv, self).__init__()
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.done_threshold = done_threshold

        # Define action and observation space
        self.action_space = [spaces.Discrete(action_size) for _ in range(num_agents)]
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32) for _ in range(num_agents)]

        self.reset()

    def reset(self,seed):
        self.states = [np.random.rand(self.state_size) for _ in range(self.num_agents)]
        return self.states

    def step(self, actions):
        rewards = []
        next_states = []
        done = [False, False]

        for i in range(self.num_agents):
            # Example transition logic
            next_state = self.states[i] + np.random.randn(self.state_size) * 0.1
            reward = -np.sum(np.square(actions[i] - 0.5))  # Example reward function
            rewards.append(reward)
            next_states.append(next_state)

            # Check if any agent's state exceeds the done threshold
            if np.any(next_state > self.done_threshold):
                done = [True, True]

        self.states = next_states
        return next_states, rewards, done, {}


class actor(Model):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(state_dim,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(action_dim, activation='softmax'))
    
    def __call__(self,x):
        x = self.model(x)
        return x


class critic(Model):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.model = Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(state_dim+action_dim,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(action_dim))
    
    def __call__(self,x,a):
        cat=tf.concat([x,a],axis=-1)
        x=self.model(cat)
        return x


class DDPG(RL):
    def __init__(self,hidden_dim,sigma,gamma,tau):
        super().__init__()
        self.env=MultiAgentEnv()
        state_dim=self.env.observation_space[0].shape[0]
        action_dim=self.env.action_space[0].n
        self.actor=[actor(state_dim,hidden_dim,action_dim) for _ in range(self.env.num_agents)]
        self.critic=[critic(state_dim,hidden_dim,action_dim) for _ in range(self.env.num_agents)]
        self.target_actor=[actor(state_dim,hidden_dim,action_dim) for _ in range(self.env.num_agents)]
        self.target_critic=[critic(state_dim,hidden_dim,action_dim) for _ in range(self.env.num_agents)]
        [assign_param(self.target_actor[i].weights,self.actor[i].weights) for i in range(self.env.num_agents)]
        [assign_param(self.target_critic[i].weights,self.critic[i].weights) for i in range(self.env.num_agents)]
        self.param=[[self.actor[i].weights for i in range(self.env.num_agents)],[self.critic[i].weights for i in range(self.env.num_agents)]]
        self.sigma=sigma
        self.gamma=gamma
        self.tau=tau
    
    def action(self,s,i):
        return self.actor[i](s)
    
    def reward_done_func_ma(self,r,d):
        return tf.reduce_mean(r),all(d)
    
    def loss(self,s,a,next_s,r,d,i_agent):
        next_q_value=self.target_critic[i_agent](next_s,self.target_actor[i_agent](next_s))
        q_target=tf.cast(r,'float32')+self.gamma*next_q_value*(1-tf.cast(d,'float32'))
        actor_loss=-tf.reduce_mean(self.critic[i_agent](s[:,i_agent],self.actor[i_agent](s[:,i_agent])))
        critic_loss=tf.reduce_mean((self.critic[i_agent](s,a)-q_target)**2)
        return [actor_loss,critic_loss]
    
    def __call__(self,s,a,next_s,r,d):
        total_actor_loss=0
        total_critic_loss=0
        for i_agent in range(self.env.num_agents):
            actor_loss,critic_loss=self.loss(s,a,next_s,r,d,i_agent)
            total_actor_loss+=actor_loss
            total_critic_loss+=critic_loss
        return total_actor_loss+total_critic_loss
    
    def update_param(self):
        for i in range(self.env.num_agents):
            for target_param,param in zip(self.target_actor[i].weights,self.actor[i].weights):
                target_param.assign(target_param*(1.0-self.tau)+param*self.tau)
            for target_param,param in zip(self.target_critic[i].weights,self.critic[i].weights):
                target_param.assign(target_param*(1.0-self.tau)+param*self.tau)
        return