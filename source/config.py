class Config:

    def __init__(self):
        self.EPS_START = 1.0         # Initial epsilon value (explore) 
        self.EPS_END = 0.01          # Last epsilon value  (exploit)
        self.EPS_DECAY = 0.99995     # Noise decay rate
        self.BUFFER_SIZE = int(1e6)  # replay buffer size
        self.BATCH_SIZE = 128        # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.TAU = 1e-3              # for soft update of target parameters
        self.LR_ACTOR = 1e-4         # learning rate of the actor
        self.LR_CRITIC = 1e-3        # learning rate of the critic
        self.WEIGHT_DECAY = 0        # L2 weight decay
