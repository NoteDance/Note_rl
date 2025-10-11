import tensorflow as tf
import torch
import numpy as np


class pr:
    def __init__(self):
        self.ratio=None
        self.TD=None
        self.index=None
        self.PPO=False
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,lambda_,alpha,batch):
        if self.PPO:
            try:
                scores=self.lambda_*self.TD+(1.0-self.lambda_)*tf.abs(self.ratio-1.0)
                prios=tf.pow(scores+1e-7,alpha)
                p=prios/tf.reduce_sum(prios)
            except Exception:
                scores=self.lambda_*self.TD+(1.0-self.lambda_)*torch.abs(self.ratio-1.0)
                prios=torch.pow(scores+1e-7,alpha)
                p=prios/torch.sum(prios)
        else:
            try:
                prios=(self.TD+1e-7)**alpha
                p=prios/tf.reduce_sum(prios)
            except Exception:
                prios=(self.TD+1e-7)**alpha
                p=prios/torch.sum(prios)
        self.index=np.random.choice(np.arange(len(state_pool)),size=[batch],p=p.numpy(),replace=False)
        try:
            self.batch.assign(batch)
        except Exception:
            self.batch=batch
        return state_pool[self.index],action_pool[self.index],next_state_pool[self.index],reward_pool[self.index],done_pool[self.index]
    
    
    def update(self,TD=None,ratio=None):
        if self.PPO:
            if TD is not None:
                TD=tf.cast(TD,tf.float32)
                ratio=tf.cast(ratio,tf.float32)
                self.TD_[:self.batch].assign(TD)
                self.ratio_[:self.batch].assign(ratio)
            else:
                self.ratio[self.index]=self.ratio_[:self.batch]
                try:
                    self.TD[self.index]=tf.abs(self.TD_[:self.batch])
                except Exception:
                    self.TD[self.index]=torch.abs(self.TD_[:self.batch])
        else:
            if TD is not None:
                TD=tf.cast(TD,tf.float32)
                self.TD_[:self.batch].assign(TD)
            else:
                try:
                    self.TD[self.index]=tf.abs(self.TD_[:self.batch])
                except Exception:
                    self.TD[self.index]=torch.abs(self.TD_[:self.batch])
        return
