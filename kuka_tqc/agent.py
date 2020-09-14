import os
import numpy as np
import sys
import torch
import copy
from tqc_models import Actor, Critic, Decoder,  quantile_huber_loss_f
import torch.nn as nn
import torch.nn.functional as F



# Building the whole Training Process into a class

class TQC(object):
    def __init__(self, state_dim, action_dim, actor_input_dim, args):
        input_dim = [args.history_length, args.size, args.size]
        self.actor = Actor(state_dim, action_dim, args).to(args.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), args.lr_actor)        
        self.critic = Critic(state_dim, action_dim, args).to(args.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), args.lr_critic)
        self.target_critic = Critic(state_dim, action_dim, args).to(args.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.decoder = Decoder(args).to(args.device)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), args.lr_decoder)
        self.target_decoder = Decoder(args).to(args.device)
        self.target_decoder.load_state_dict(self.decoder.state_dict())
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.tau = args.tau 
        self.device = args.device
        self.write_tensorboard = False
        self.top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets
        self.n_nets = args.n_nets
        self.top_quantiles_to_drop_per_net = args.top_quantiles_to_drop_per_net
        self.target_entropy = args.target_entropy 
        self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=args.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr_alpha)
        self.total_it = 0
        self.step = 0
        self.beta = 0.5

    def update_beta(self, replay_buffer, writer, total_timesteps):
        obs, action, reward, next_obs, not_done, obs_aug, obs_next_aug = replay_buffer.get_last_k_trajectories()  # not done are 0 if episodes ends
        store_batch = []
        # create a R_i for the k returns from the buffer
        i = 0
        obs = obs.div_(255)
        for idx in range(obs.shape[0]):
            if not_done[idx][0] == 0:
                i = 0
            R_i =  self.discount**i * reward[idx][0]
            # print("Ri", R_i)
            store_batch.append((obs[idx], action[idx], R_i))
            i += 1
        delta = 0
        # print(store_batch)
        for b in store_batch:
            s ,a , r = b[0], b[1], b[2].data.item()
            a = a.unsqueeze(0) 
            s = s.unsqueeze(0)
            r = torch.Tensor(np.array([r]))
            r = r.unsqueeze(1).to(self.device)
            state_aug = self.decoder.create_vector(s)
            next_z = self.critic(state_aug.detach(), a.detach())
            Q = 0
            for net in next_z[0]:
                Q += torch.mean(net).data.item()
            Q *= 1. / self.n_nets
            delta +=  Q
        delta *= (1. / len(store_batch))
        
        writer.add_scalar('delta', delta, total_timesteps)
        
        dif  = self.beta - delta
        writer.add_scalar('dif', dif, total_timesteps)
        self.beta = max(0., min(dif, 1))
        writer.add_scalar('beta', self.beta, total_timesteps)
        # compute how many quntile to drop if beta is greate 0.5 
        # if beta gets closer to 1 drop more qantuile
        if self.beta > 0.5 and self.top_quantiles_to_drop_per_net > 0:
            self.top_quantiles_to_drop_per_net -= 1
        if self.beta < 0.5 and self.top_quantiles_to_drop_per_net < 5:
            self.top_quantiles_to_drop_per_net += 1
        writer.add_scalar('drop-qunantile', self.top_quantiles_to_drop_per_net , total_timesteps)
        self.top_quantiles_to_drop = self.top_quantiles_to_drop_per_net * self.n_nets

    
    def train(self, replay_buffer, writer, iterations):
        self.step += 1
        if self.step % 1000 == 0:
            self.write_tensorboard = 1 - self.write_tensorboard
        for it in range(iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memoy
            sys.stdout = open(os.devnull, "w")
            obs, action, reward, next_obs, not_done, obs_aug, obs_next_aug = replay_buffer.sample(self.batch_size)
            sys.stdout = sys.__stdout__
            # for augment 1
            obs = obs.div_(255)
            next_obs = next_obs.div_(255)
            state = self.decoder.create_vector(obs)
            detach_state = state.detach()
            next_state = self.target_decoder.create_vector(next_obs)
            # for augment 2
            
            obs_aug = obs_aug.div_(255)
            next_obs_aug = obs_next_aug.div_(255)
            state_aug = self.decoder.create_vector(obs_aug)
            detach_state_aug = state_aug.detach()
            next_state_aug = self.target_decoder.create_vector(next_obs_aug)
            
            alpha = torch.exp(self.log_alpha)
            with torch.no_grad(): 
                # Step 5: Get policy action
                new_next_action, next_log_pi =  self.actor(next_state)
                
                # compute quantile at next state
                next_z = self.target_critic(next_state, new_next_action)
                sorted_z, _ = torch.sort(next_z.reshape(self.batch_size, -1))
                sorted_z_part = sorted_z[:,:self.quantiles_total - self.top_quantiles_to_drop]
                target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)
                         
                # again for augment
                new_next_action_aug, next_log_pi_aug = self.actor(next_state_aug)
                next_z_aug = self.target_critic(next_state_aug, new_next_action_aug)
                sorted_z_aug, _ = torch.sort(next_z_aug.reshape(self.batch_size, -1))
                sorted_z_part_aug = sorted_z_aug[:,:self.quantiles_total - self.top_quantiles_to_drop]
                target_aug = reward + not_done * self.discount * (sorted_z_part_aug - alpha * next_log_pi_aug)
             
            target = (target + target_aug) / 2.
            #---update critic
            cur_z = self.critic(state, action)
            #print("curz shape", cur_z.shape)
            #print("target shape", target.shape)
            critic_loss = quantile_huber_loss_f(cur_z, target, self.device)
            
            # for augment
            cur_z_aug = self.critic(state_aug, action)
            critic_loss += quantile_huber_loss_f(cur_z_aug, target, self.device)
            self.critic_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            critic_loss.backward()
            self.decoder_optimizer.step()
            self.critic_optimizer.step()
        
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.decoder.parameters(), self.target_decoder.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            #---Update policy and alpha
            new_action, log_pi = self.actor(detach_state)
            alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
            actor_loss = (alpha * log_pi - self.critic(detach_state, new_action).mean(2).mean(1, keepdim=True)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.total_it +=1
    
    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.div_(255)
        state = self.decoder.create_vector(obs.unsqueeze(0))
        return self.actor.select_action(state)

                

    def quantile_huber_loss_f(self, quantiles, samples):
        pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
        n_quantiles = quantiles.shape[2]
        tau = torch.arange(n_quantiles, device=self.device).float() / n_quantiles + 1 / 2 / n_quantiles
        loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
        return loss

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
                
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor) 
