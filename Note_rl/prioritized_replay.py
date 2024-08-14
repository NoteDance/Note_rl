import numpy as np


class pr:
    def __init__(self):
        self.TD=np.array(0)
        self.index=None
    
    
    def sample(self,state_pool,action_pool,next_state_pool,reward_pool,done_pool,epsilon,alpha,batch):
        p=(self.TD[1:]+epsilon)**alpha/np.sum((self.TD[1:]+epsilon)**alpha)
        self.index=np.random.choice(np.arange(len(state_pool),dtype=np.int8),size=[batch],p=p)
        return state_pool[self.index],action_pool[self.index],next_state_pool[self.index],reward_pool[self.index],done_pool[self.index]
    
    
    def update_TD(self,TD):
        for i in range(len(self.index)):
            self.TD[1:][self.index[i]]=TD[i]
        return