import random
import numpy as np
import sys
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

from test.env import hitorstandcontinuous
from test.policy import DQN
from test.components import noisebuffer
from test.buffer import ReplayMemory
from test.components import Transition


class DPDQ:
    def __init__(self,
                 seed,
                 sigma,
                 BATCH_SIZE=128,
                 GAMMA=0.999,
                 EPS_START=0.9,
                 EPS_END=0.05,
                 EPS_DECAY=200,
                 TARGET_UPDATE=10,
                 device="cpu",
                 EPISODE_NUM=100
                 ):

        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.episode_start = EPS_START
        self.episode_end = EPS_END
        self.episode_decay = EPS_DECAY
        self.update_rate = TARGET_UPDATE
        self.steps_done = 0

        self.num_episodes = EPISODE_NUM

        self.seed = np.random.seed(seed)

        self.device = device

        self.env = hitorstandcontinuous()

        # TODO: discrete
        self.action_size = self.env.action_space.n
        # TODO continuous
        # self.action_size = self.env.action_space.shape[0]

        self.noise_buffer = noisebuffer(2, sigma)
        self.policy = DQN(sigma=sigma,
                          action_space=self.action_size,
                          noisebuffer=self.noise_buffer).to(self.device)
        self.target_net = DQN(sigma=sigma,
                              action_space=self.action_size,
                              noisebuffer=self.noise_buffer).to(self.device)
        self.target_net.load_state_dict(self.policy.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy.parameters())

        self.memory = ReplayMemory(10000)

        self.episode_durations = []

    def get_seed(self):
        return self.seed

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.episode_end + (self.episode_start - self.episode_end) * \
                        math.exp(-1. * self.steps_done / self.episode_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest value for column of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        # import pdb; pdb.set_trace()
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def render(self):
        # plt.figure(2)
        # plt.clf()
        # durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        # plt.title('Training...')
        # plt.xlabel('Episode')
        # plt.ylabel('Duration')
        # plt.plot(durations_t.numpy())
        # # Take 100 episode averages and plot them too
        # if len(durations_t) >= 100:
        #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        #     means = torch.cat((torch.zeros(99), means))
        #     plt.plot(means.numpy())
        #
        # plt.pause(0.001)  # pause a bit so that plots are updated
        t = [i + 1 for i in range(len(self.episodic_rewards))]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, self.episodic_rewards)
        ax.set_xlabel('time')
        ax.set_ylabel('s1 and s2')

    def train(self):
        self.episodic_rewards = []
        episode_durations = []
        for i_episode in range(self.num_episodes):
            # if i_episode % 10 == 0:
            # print(i_episode)
            # Initialize the environment and state
            state = torch.Tensor(self.env.reset()).unsqueeze(0)
            total_reward = 0
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                if not done:
                    next_state = torch.Tensor(next_state).unsqueeze(0)
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
                # import pdb; pdb.set_trace()
                total_reward += float(reward.squeeze(0).data)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    episode_durations.append(t + 1)
                    # plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.update_rate == 0:
                self.target_net.load_state_dict(self.policy.state_dict())
            self.episodic_rewards.append(total_reward)
            self.policy.nb.reset()
            self.target_net.nb.reset()
        return self.episodic_rewards


if __name__ == "__main__":
    seed = int(sys.argv[1])
    sigma = float(sys.argv[2])

    plt.ion()

    dpq = DPDQ(seed,
               sigma,
               BATCH_SIZE=128,
               GAMMA=0.999,
               EPS_START=0.9,
               EPS_END=0.05,
               EPS_DECAY=200,
               TARGET_UPDATE=10,
               device="cpu")

    torch.manual_seed(seed)

    episodic_rewards = dpq.train()
    # dpq.render()

    print('Complete')
    print(episodic_rewards)
    with open('dpql.txt', 'a') as fw:
        for rr in episodic_rewards:
            fw.write(str(rr))
            fw.write(' ')
        fw.write('\n')