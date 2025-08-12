import numpy as np


class pr:
    def __init__(self):
        self.ratio=None
        self.TD=None
        self.index=None
        self.PPO=False
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,epsilon,lambda_,alpha,batch):
        if self.PPO:
            scores=self.lambda_*self.TD+(1.0-self.lambda_)*np.abs(self.ratio-1.0)
            prios=np.pow(scores+epsilon,alpha)
            p=prios/np.sum(prios)
        else:
            prios=(self.TD+epsilon)**alpha
            p=prios/np.sum(prios)
        self.index=np.random.choice(np.arange(len(state_pool)),size=[batch],p=p,replace=False)
        return state_pool[self.index],action_pool[self.index],next_state_pool[self.index],reward_pool[self.index],done_pool[self.index]
    
    
    def update(self,TD=None,ratio=None):
        if self.PPO:
            if TD is not None:
                self.TD_=TD
                self.ratio_=ratio
            else:
                self.ratio[self.index]=self.ratio_
                self.TD[self.index]=np.abs(self.TD_)
        else:
            if TD is not None:
                self.TD_=TD
            else:
                self.TD[self.index]=np.abs(self.TD_)
        return
