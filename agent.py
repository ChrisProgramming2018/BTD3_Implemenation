import torch
from models import Actor, Critic
import torch.nn as nn
import torch.nn.functional as F



class TD3(object):
    """ TD3 plus the Ensemble of Critics as an agent object
    to act and update the networkweights, save and laod the weights
    """
    def __init__(self, state_dim, action_dim, max_action, args):
        self.actor = Actor(state_dim, action_dim, max_action).to(args.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(args.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), args.lr_actor)
        self.critic = Critic(state_dim, action_dim).to(args.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), args.lr_critic)
        self.target_critic = Critic(state_dim, action_dim).to(args.device)
        self.target_critic.load_state_dict(self.target_critic.state_dict())
        self.max_action = max_action
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.tau = args.tau 
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.device = args.device
        self.step = 0
        self.beta = 0.5
    
    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def compute_beta(self, replay_buffer):
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.get_last_k_trajectories()
        #state = torch.Tensor(batch_states).to(self.device)
        #action = torch.Tensor(batch_actions).to(self.device)
        store_batch = []
        # create a R_i for the k returns from the buffer
        i = 0
        for idx in range(len(batch_states)):
            if batch_dones[idx][0] != 0:
                i = 0
            R_i =  self.discount**i * batch_rewards[idx][0]
            store_batch.append((batch_states[idx], batch_actions[idx], R_i))
            i += 1
        #print(store_batch)
        delta = 0
        for b in store_batch:
            s ,a ,r = b[0], b[1], b[2]
            a = torch.Tensor(a).to(self.device).unsqueeze(0) 
            s = torch.Tensor(s).to(self.device).unsqueeze(0) 
            Q1, Q2 = self.critic(a.detach(), s.detach())
            Q = 0.5 * (Q1 + Q2)
            delta += Q -r 
        delta *= (1. / len(batch_states))
        # print("delta ", delta)
        self.beta = torch.clamp(self.beta - delta, 0., 1.).item()
        # print("beta ", self.beta)

            




    
    def train(self, replay_buffer, writer, iterations):
        """ Update function for the networkweights of the Actor and Critis 
            current and Target by useing the 3 new features of the TD3 paper
            to the DDPG implementation
            1. Delay the policy Updates 
            2. Two crtitc networks take the min Q value
            3. Target Policy Smoothing
            Own use an Ensemble Approach of delayed updated critics 
        
        """
        self.step += 1
        
        for it in range(iterations):
            
            # Step 1: Sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(self.batch_size)
            # convert the numpy arrays to tensors object 
            # if cuda is available send data to gpu
            state = torch.Tensor(batch_states).to(self.device)
            next_state = torch.Tensor(batch_next_states).to(self.device)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)
            
            # Step 2: use the  Target Actor  to create the action of the next
            # state (part of the TD-Target 
            next_action = self.actor_target(next_state)
            
            # Step 3: Add Gaussian noise to the action for exploration 
            # clip the action value in case its outside the boundaries 
            noise = torch.Tensor(batch_actions).data.normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            # Step 4: Use the differenet Target Critis (delayed update)
            # and min of two  Critic from TD3 to create the different Q Targets
            # compute the average of all Q Targets to get a single value
            target_Q1, target_Q2 = self.target_critic(next_state, next_action) 
            target_Q_min = torch.min(target_Q1, target_Q2)
            
            target_Q_1 = reward + ((1 - done) * self.discount * target_Q1).detach()
            target_Q_2 = reward + ((1 - done) * self.discount * target_Q2).detach()
            # Step 5: Create the update based on the bellman  equation 
            target_Q_min = reward + ((1 - done) * self.discount * target_Q_min).detach()

            y1 = (self.beta * (target_Q1)) + ((1 - self.beta) * target_Q_min)
            y2 = (self.beta * (target_Q2)) + ((1 - self.beta) * target_Q_min)

            # Step 6: Use the critic compute the Q estimate for current state and action
            current_Q1, current_Q2 = self.critic(state, action) 
            
            # Step 7: Compute the critc loss with the mean squard error
            # loss function 
            critic_loss = F.mse_loss(current_Q1, y1.detach()) + F.mse_loss(current_Q2, y2.detach())
            writer.add_scalar('critic_loss', critic_loss , self.step)
            

            # Step 8: Backpropagate this Critic loss and update the parameters
            # of the two Critic models use the adam optimizer
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Step 9: Delayed update og Actor model
            if it % self.policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                writer.add_scalar('actor_loss', actor_loss , self.step)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Step 10: we update the weights of the Critic target by polyak averaging
                # hyperparameter tau determines the combination of current and
                # target weights
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
     
                
    
    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
    
    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
