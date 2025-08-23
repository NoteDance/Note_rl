import tensorflow as tf
from Note_rl.RL import RL
from Note_rl.assign_param import assign_param
from keras.models import Sequential
from keras import Model
import gym


class actor(Model):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.model = Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(state_dim,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(action_dim))
    
    def __call__(self,x):
        x=self.model(x)
        return tf.nn.softmax(x)


class critic(Model):
    def __init__(self,state_dim,hidden_dim):
        super().__init__()
        self.model = Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_dim, input_shape=(state_dim,), activation='relu'))
        self.model.add(tf.keras.layers.Dense(1))
    
    def __call__(self,x):
        x=self.model(x)
        return x


class Controller(Model):
    def __init__(self, hidden=32, temp=10.0):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden, activation='relu')
        self.fc2 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.max_w = None
        self.temp = temp

    def __call__(self, features):
        x = self.fc1(features)
        alpha = self.fc2(x)
        w = alpha * self.max_w
        return tf.squeeze(w, axis=-1)
    
    
class PPO(RL):
    def __init__(self,state_dim,hidden_dim,action_dim,clip_eps,alpha,temp=10.0):
        super().__init__()
        self.actor=actor(state_dim,hidden_dim,action_dim)
        self.actor_old=actor(state_dim,hidden_dim,action_dim)
        self.controller = Controller()
        assign_param(self.actor_old.weights,self.actor.weights)
        self.critic=critic(state_dim,hidden_dim)
        self.clip_eps=clip_eps
        self.alpha=alpha
        self.temp = temp
        self.param=[self.actor.weights,self.critic.weights,self.controller.weights]
        self.env=gym.make('CartPole-v0')
    
    def action(self,s):
        return self.actor_old(s)
    
    def window_size_fn(self):
        ratio_score = tf.reduce_sum(tf.abs(self.prioritized_replay.ratio-1.0))
        td_score = tf.reduce_sum(self.prioritized_replay.TD)
        scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * tf.abs(self.prioritized_replay.ratio - 1.0)
        weights = tf.pow(scores + 1e-7, self.alpha)
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        features = tf.reshape([ratio_score, td_score, ess, len(self.prioritized_replay.ratio)], (1,4))
        features = (features - tf.reduce_min(features)) / (tf.reduce_max(features) - tf.reduce_min(features) + 1e-8)
        return self.controller(features)
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        action_prob=tf.gather(self.actor(s),a,axis=1,batch_dims=1)
        action_prob_old=tf.gather(self.actor_old(s),a,axis=1,batch_dims=1)
        ratio=action_prob/action_prob_old
        value=self.critic(s)
        value_tar=tf.cast(r,'float32')+0.98*self.critic(next_s)*(1-tf.cast(d,'float32'))
        TD=value_tar-value
        sur1=ratio*TD
        sur2=tf.clip_by_value(ratio,clip_value_min=1-self.clip_eps,clip_value_max=1+self.clip_eps)*TD
        clip_loss=-tf.math.minimum(sur1,sur2)
        entropy=action_prob*tf.math.log(action_prob+1e-8)
        clip_loss=clip_loss-self.alpha*entropy
        self.controller.max_w = len(self.prioritized_replay.ratio)
        ratio_score = tf.reduce_sum(tf.abs(self.prioritized_replay.ratio-1.0))
        td_score = tf.reduce_sum(self.prioritized_replay.TD)
        score = ratio_score + td_score
        scores = self.lambda_ * self.prioritized_replay.TD + (1.0-self.lambda_) * tf.abs(self.prioritized_replay.ratio - 1.0)
        weights = tf.pow(scores + 1e-7, self.alpha)
        p = weights / (tf.reduce_sum(weights))
        ess = 1.0 / (tf.reduce_sum(p * p))
        features = tf.reshape([ratio_score, td_score, ess, len(self.prioritized_replay.ratio)], (1,4))
        features = (features - tf.reduce_min(features)) / (tf.reduce_max(features) - tf.reduce_min(features) + 1e-8)
        w = self.controller(features)
        idx = tf.cast(tf.range(len(self.prioritized_replay.ratio), w.dtype))
        m = tf.sigmoid((idx - w) / self.temp)
        controller_loss = tf.reduce_mean(m * score)
        self.prioritized_replay.update(TD,ratio)
        return tf.reduce_mean(clip_loss)+tf.reduce_mean((TD)**2)+controller_loss
    
    def update_param(self):
        assign_param(self.actor_old.weights, self.actor.weights)
        return


class PPO_(RL):
    def __init__(self,state_dim,hidden_dim,action_dim,clip_eps,alpha):
        super().__init__()
        self.actor=actor(state_dim,hidden_dim,action_dim)
        self.actor_old=actor(state_dim,hidden_dim,action_dim)
        self.controller = Controller()
        assign_param(self.actor_old.weights,self.actor.weights)
        self.critic=critic(state_dim,hidden_dim)
        self.clip_eps=clip_eps
        self.alpha=alpha
        self.param=[self.actor.weights,self.critic.weights]
        self.env=gym.make('CartPole-v0')
    
    def action(self,s):
        return self.actor_old(s)
    
    def window_size_fn(self):
        return self.adjust_window_size()
    
    def __call__(self,s,a,next_s,r,d):
        a=tf.expand_dims(a,axis=1)
        action_prob=tf.gather(self.actor(s),a,axis=1,batch_dims=1)
        action_prob_old=tf.gather(self.actor_old(s),a,axis=1,batch_dims=1)
        ratio=action_prob/action_prob_old
        value=self.critic(s)
        value_tar=tf.cast(r,'float32')+0.98*self.critic(next_s)*(1-tf.cast(d,'float32'))
        TD=value_tar-value
        sur1=ratio*TD
        sur2=tf.clip_by_value(ratio,clip_value_min=1-self.clip_eps,clip_value_max=1+self.clip_eps)*TD
        clip_loss=-tf.math.minimum(sur1,sur2)
        entropy=action_prob*tf.math.log(action_prob+1e-8)
        clip_loss=clip_loss-self.alpha*entropy
        self.prioritized_replay.update(TD,ratio)
        return tf.reduce_mean(clip_loss)+tf.reduce_mean((TD)**2)
    
    def update_param(self):
        assign_param(self.actor_old.weights, self.actor.weights)
        return
