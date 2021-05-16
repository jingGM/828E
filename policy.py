import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self,
                 noisebuffer,
                 action_space,
                 sigma=0.4,
                 hidden=16,
                 ):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(1, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, action_space)
        self.sigma = sigma
        self.nb = noisebuffer

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = self.head(x)
        if self.sigma > 0:
            eps = [self.nb.sample(float(state)) for state in s]
            eps = torch.Tensor(eps)
            return x + eps
        else:
            return x
