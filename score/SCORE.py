import copy
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from rlkit.core import logger, eval_util
from rlkit.core.eval_util import create_stats_ordered_dict
from collections import OrderedDict

def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, spectral_norm=False):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        if spectral_norm:
            self.l2 = nn.utils.spectral_norm(nn.Linear(256, 256))
        else:
            self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, spectral_norm=False):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        if spectral_norm:
            self.l2 = nn.utils.spectral_norm(nn.Linear(256, 256))
        else:
            self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

class SCORE(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount=0.99,
        tau=0.005,
        lr=3e-4,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
        num_ensemble=5,
        beta=0.2,
        bc_decay=0.98,
        spectral_norm=False,
    ):

        self.actor = Actor(state_dim, action_dim, max_action, spectral_norm).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.L_critic, self.L_critic_target, self.L_critic_optimizer = [], [], []
        for _ in range(num_ensemble):
            critic = Critic(state_dim, action_dim, spectral_norm).to(device)
            critic_target = copy.deepcopy(critic)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
            self.L_critic.append(critic)
            self.L_critic_target.append(critic_target)
            self.L_critic_optimizer.append(critic_optimizer)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.discount = discount
        self.tau = tau
        self.lr = lr
        self.expl_noise =  expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.num_ensemble = num_ensemble
        self.beta = beta
        self.bc = 1.0
        self.bc_decay = bc_decay

        self._num_update_steps = 0
        self._num_critic_update_steps = 0
        self._num_actor_update_steps = 0
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return get_numpy(self.actor(state)).flatten()

    def get_snapshot(self):
        return dict(
            actor=self.actor,
            L_critic=self.L_critic,
            actor_target=self.actor_target,
            L_critic_target=self.L_critic_target,
        )

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def get_std(self, obs, policy_action, target=True):
        L_target_Q = []
        mean_Q, var_Q, min_Q = 0.0, 0.0, torch.ones(1).to(self.device)*1000.0
        for en_index in range(self.num_ensemble):
            if target:
                target_Q = self.L_critic_target[en_index](obs, policy_action)               
            else:
                target_Q = self.L_critic[en_index](obs, policy_action)
            L_target_Q.append(target_Q)
            mean_Q += target_Q / self.num_ensemble
            min_Q = torch.min(min_Q, target_Q)

        mean_Q = mean_Q.detach()
        for target_Q in L_target_Q:
            var_Q += (target_Q - mean_Q)**2
        var_Q = var_Q / self.num_ensemble
        std_Q = torch.sqrt(var_Q)
        return mean_Q, std_Q, min_Q

    def mask_mean(self, loss, mask):
        return (loss * mask).sum() / (mask.sum()+1) # use +1 to avoid divided by 0

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def train(self, replay_buffer, iterations, batch_size=256):

        for _ in range(iterations):
            # Sample replay buffer 
            state, action, next_state, reward, not_done, masks = replay_buffer.sample(batch_size)

            """
            Initialze Statistics
            """
            actor_loss, q_loss, bc_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            std_Q_new, new_current_Q, new_action = torch.zeros(1), torch.zeros(1), torch.zeros(1)

            """
            Critic Training
            """
            L_target_Q = []
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                new_next_action = (
                    self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the uncertainty
                _, std_Q, _ = self.get_std(state, action, target=False)
                _, std_Q_pi, _ = self.get_std(next_state, new_next_action, target=True)
                lmbda = 1/std_Q.abs().mean().detach()

                # Compute the target Q value
                for en_idx in range(self.num_ensemble):
                    target_Q_ = self.L_critic_target[en_idx](next_state, new_next_action)                  
                    target_Q = reward + not_done * self.discount * target_Q_ - self.beta*std_Q
                    L_target_Q.append(target_Q)

            # Get current Q estimates
            for en_idx in range(self.num_ensemble):
                mask = masks[:,en_idx].reshape(-1, 1)
                current_Q = self.L_critic[en_idx](state, action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q, L_target_Q[en_idx], reduce=False)
                critic_loss = self.mask_mean(critic_loss, mask)

                # Optimize the critic
                self.L_critic_optimizer[en_idx].zero_grad()
                critic_loss.backward()
                self.L_critic_optimizer[en_idx].step()
            self._num_critic_update_steps += 1
            
            """
            (Delayed) Actor Training 
            """
            if self._num_update_steps % self.policy_freq == 0:

                # Compute actor loss
                new_action = self.actor(state)
                _, std_Q_new, min_Q = self.get_std(state, new_action, target=False)
                new_current_Q = min_Q

                lmbda = self.alpha/new_current_Q.abs().mean().detach()
                q_loss = new_current_Q
                bc_loss = F.mse_loss(new_action, action, reduce=False).mean(dim=1)
                actor_loss = -lmbda * q_loss + self.bc*bc_loss.mean()
                actor_loss = actor_loss.mean()
                
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self._num_actor_update_steps += 1

                # Update the frozen target models
                for en_idx in range(self.num_ensemble):
                    for param, target_param in zip(self.L_critic[en_idx].parameters(), self.L_critic_target[en_idx].parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            self._num_update_steps += 1
            if self.bc_decay < 1 and self._num_update_steps % 10000 ==0:
                self.bc = self.bc*self.bc_decay

            """
            Save Statistics for Log
            """
            if self._need_to_update_eval_statistics:
                self._need_to_update_eval_statistics = False
                """
                Eval should set this to None.
                This way, these statistics are only computed for one batch.
                """
                # Losses
                self.eval_statistics['Critic Loss'] = np.mean(get_numpy(critic_loss))
                self.eval_statistics['Policy Loss'] = np.mean(get_numpy(actor_loss))
                self.eval_statistics['Q Loss'] = np.mean(get_numpy(q_loss.mean()))
                self.eval_statistics['BC Loss'] = np.mean(get_numpy(bc_loss.mean()))
                # Gradients
                self.eval_statistics['QF Grad'] = np.mean(get_numpy(self.L_critic[0].l3.weight.grad.mean()))
                self.eval_statistics['Policy Grad'] = np.mean(get_numpy(self.actor.l3.weight.grad.mean()))
                # Q values
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q Predictions',
                    get_numpy(current_Q),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q Targets',
                    get_numpy(target_Q),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q Curr Predictions',
                    get_numpy(new_current_Q),
                ))
                # Uncertainty
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q std',
                    get_numpy(std_Q),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q pi std',
                    get_numpy(std_Q_pi),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q new std',
                    get_numpy(std_Q_new),
                ))
                # Actions
                self.eval_statistics.update(create_stats_ordered_dict(
                    'New curr actions', 
                    get_numpy(new_action)
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'New next actions', 
                    get_numpy(new_next_action)
                ))
                # Misc
                self.eval_statistics['bc'] = self.bc
                self.eval_statistics['mask ratio'] = np.mean(get_numpy(masks))
                self.eval_statistics['Num Q Updates'] = self._num_critic_update_steps
                self.eval_statistics['Num Policy Updates'] = self._num_actor_update_steps            
                self.eval_statistics['Num Total Updates'] = self._num_update_steps


    def save(self, filename):
        for en_idx in range(self.num_ensemble):
            torch.save(self.L_critic[en_idx].state_dict(), filename + "_critic" + str(en_idx))
            torch.save(self.L_critic_optimizer[en_idx].state_dict(), filename + "_critic_optimizer" + str(en_idx))
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, log_dir):
        params = torch.load(log_dir+'/params.pkl')
        self.actor = params['actor']
        self.actor_target = copy.deepcopy(self.actor)
        self.L_critic = params['L_critic']
        self.L_critic_target = copy.deepcopy(self.L_critic)
