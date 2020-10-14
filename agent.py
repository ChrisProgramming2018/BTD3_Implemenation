import gym
from replay_buffer2 import ReplayBuffer 
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from collections import deque

class Agent():
    def __init__(self, state_size, action_size, config):
        self.action_size = action_size
        self.state_size = state_size
        self.Q = np.zeros([state_size, action_size])
        self.debug_Q = np.zeros([state_size, action_size])
        self.Q_shift = np.zeros([state_size, action_size])
        self.r = np.zeros([state_size, action_size])
        self.counter = np.zeros([state_size, action_size])
        self.gamma = config["gamma"]
        self.epsilon = 1
        self.lr = config["lr"]
        self.min_epsilon = config["min_epsilon"]
        self.max_epsilon =1
        self.episode = 15000
        self.decay = config["decay"]
        self.total_reward = 0
        self.eval_frq = 50
        self.render_env = False
        self.env = gym.make(config["env_name"])
        self.memory = ReplayBuffer((1,),(1,),config["buffer_size"], config["device"])
        self.gamma_iql = 0.99
        self.lr_sh = 0.07
        self.ratio = 1. / action_size
        self.eval_q_inverse = 50000
        self.episodes_qinverse = int(5e6)
        self.steps = 0
        pathname = ""
        tensorboard_name = str(config["locexp"]) + '/runs/' + pathname 
        self.writer = SummaryWriter(tensorboard_name)
        self.last_100_reward_errors = deque(maxlen=100) 
    
    def act(self, state, epsilon, eval_pi=False, use_debug=False):

        if np.random.random() > epsilon or eval_pi:
            action = np.argmax(self.Q[state])
            if use_debug:
                action = np.argmax(self.debug_Q[state])
        else:
            action = self.env.action_space.sample() 
        return action
    
    
    def optimize(self, state, action, reward, next_state, debug=False):
        if debug:
            max_next_state = np.max(self.debug_Q[next_state])
            td_error =  max_next_state - self.debug_Q[state, action]
            self.debug_Q[(state,action)] = self.debug_Q[(state,action)] + self.lr * (reward + self.gamma *td_error)
            return

        max_next_state = np.max(self.Q[next_state])
        td_error =  max_next_state - self.Q[state, action]
        self.Q[(state,action)] = self.Q[(state,action)] + self.lr * (reward + self.gamma *td_error)
    
    def learn(self):
        states, actions, rewards, next_states, done =  self.memory.sample(self.batch_size)
        # update Q function
        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, done):
            max_next_state = np.max(self.Q[next_state])
            td_error = self.Q[state, action] - max_next_state
            self.Q[(state,action)] = self.Q[(state,action)] + self.lr * (reward + self.gamma*  td_error)
    
    def compute_reward_loss(self, episode=10):
        """
        use the env to create the real reward and compare it to the predicted
        reward of the model
 
        """
        self.env.seed(np.random.randint(0,10))
        reward_loss = 0
        reward_list = []
        for epi in range(episode):
            state = self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.trained_Q[state])
                next_state, reward, done, _ = self.env.step(action)
                predict_reward = self.r[state, action]
                reward_list.append((reward, predict_reward))
                if done: 
                    break
        reward_loss =([abs(r[0] - r[1]) for r in reward_list]  )
        reward_loss_length = len(reward_loss)
        reward_loss = sum(reward_loss) / reward_loss_length
        self.last_100_reward_errors.append(reward_loss)
        average_loss = np.mean(self.last_100_reward_errors)
        print("average mean loss ", average_loss)
        self.writer.add_scalar('Reward_loss', reward_loss, self.steps)
        self.writer.add_scalar('Average_Reward_loss', average_loss, self.steps)
        #print(reward_loss)

    
    def invers_q(self, continue_train=False):
        
        if not continue_train:
            print("clean policy")
            self.Q = np.zeros([self.state_size, self.action_size])
        
        for epi in range(1, self.episodes_qinverse + 1):
            self.steps += 1
            text = "Inverse Episode {} \r".format(epi)
            print(text, end= "")
            if epi % self.eval_q_inverse == 0:
                self.render_env = True
                #self.eval_policy(use_expert=True, episode=1)
                #self.eval_policy(random_agent=True, episode=1)
                # self.reward_loss()
                self.eval_policy(episode=1)
                self.render_env =False
            state, action, _, next_state, _ = self.memory.sample(1)
            self.counter[state,action] +=1
            total_num = np.sum(self.counter[state,:])
            action_prob = self.counter[state] / total_num
            #print(np.sum(action_prob))
            assert(np.isclose(np.sum(action_prob),1))
            # update Q shift 
            Q_shift_target = self.lr_sh *(self.gamma_iql * np.max(self.Q[next_state]))
            self.Q_shift[state, action] = (1 - self.lr_sh) * self.Q_shift[state, action] + Q_shift_target
            # compute n a 
            n_a = action_prob[0][0][action] - self.Q_shift[state, action]
            
            # update reward function
            for i in range(20):
                self.update_r(state, action, n_a, action_prob)
            self.debug_train()
            # update Q function
            # self.update_q(state, action, next_state)

    def update_q(self, state, action, next_state):
        q_old = (1 - self.lr) * self.Q[state, action]
        q_new = self.lr *(self.r[state, action] + self.gamma_iql * np.max(self.Q[next_state]))
        self.Q[state, action] = q_old + q_new
        
    def update_r(self, state, action, n_a, action_prob):
        r_old = (1 - self.lr) * self.r[state, action]
        part1 = n_a
        print("part1", n_a)
        part2 = self.ratio * self.sum_over_action(state, action, action_prob)
        r_new = self.lr * (part1 + part2)
        print("r old ", r_old)
        print("r_new", r_new)
        self.r[state, action] = r_old + r_new       
    
    def sum_over_action(self, state, a, action_prob):
        res = 0
        for b in range(self.action_size):
            if b == a:
                continue
            res += self.r[state, b] - self.compute_n_a(state, b, action_prob)
        return res
    
    def compute_n_a(self, state, a, action_prob):
        return action_prob[0][0][a] - self.Q_shift[state, a]
    
    def eval_policy(self, random_agent=False, use_expert=False, use_debug=False, episode=10):
        if use_expert:
            self.load_q_table()
        total_steps = 0
        total_reward = 0
        total_penetlies = 0
        for i_episode in range(1, episode + 1):
            score = 0
            steps = 0
            state = self.env.reset()
            done  = False
            penelty = 0
            while not done:
                steps += 1
                action = self.act(state, 0, True)
                if use_expert:
                    action = np.argmax(self.trained_Q[state])
                if random_agent:
                    action = self.env.action_space.sample() 
                if use_debug:
                    action = np.argmax(self.debug_Q[state])
                
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                if self.render_env:
                    self.env.render()
                    time.sleep(0.1)
                score += reward
                if reward == -10:
                    penelty += 1
                if done or steps == 100:
                    total_steps += steps
                    total_reward += score
                    total_penetlies += penelty
                    break
        if self.render_env:
            self.env.close()
        aver_steps = total_steps / episode
        average_reward = total_reward / episode
        aver_penelties = total_penetlies / episode
        if random_agent:
            print("Random Eval avge steps {} average reward  {:.2f}  average penelty {} ".format(aver_steps, average_reward, aver_penelties))
        else:    
            print("Eval avge steps {} average reward  {:.2f}  average penelty {} ".format(aver_steps, average_reward, aver_penelties))
            self.writer.add_scalar('Eval_Average_steps', aver_steps, self.steps)
            self.writer.add_scalar('Eval_Average_reward', average_reward, self.steps)
            self.writer.add_scalar('Eval_Average_penelties', aver_penelties, self.steps)
       
    def save_q_table(self, filename="policy"):
        with open(filename + '/Q.npy', 'wb') as f:
            np.save(f, self.Q)

    def load_q_table(self, filename="policy"):
        with open(filename + '/Q.npy', 'rb') as f:
            self.Q = np.load(f)

        self.trained_Q = self.Q
        self.memory.idx = 30000

    def create_expert_policy(self):
        self.load_q_table()
        self.trained_Q = self.Q
        for i_episode in range(1, self.episode + 1):
            text = "create Buffer {} of {}\r".format(i_episode, self.episode)
            print(text, end=" ")
            state = self.env.reset()
            done  = False
            while not done:
                action = self.act(state, 0, True)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                self.memory.add(state, action, reward, next_state, done, done)



    def train(self):
      
        total_timestep = 0
        for i_episode in range(1, self.episode + 1):
            score = 0
            state = self.env.reset()
            done  = False
            steps = 0
            while not done:
                self.steps +=1
                steps += 1
                total_timestep += 1
                action = self.act(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.optimize(state, action, reward, next_state)
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay* i_episode)
                
                if done:
                    break
                state = next_state
            
            if i_episode % self.eval_frq == 0:
                self.eval_policy()
            
            self.total_reward += score
            average_reward = self.total_reward / i_episode
            print("Episode {} Reward {:.2f} Average Reward {:.2f} steps {}  epsilon {:.2f}".format(i_episode, score, average_reward, steps, self.epsilon))
            self.writer.add_scalar('Average_reward', average_reward, self.steps)
            self.writer.add_scalar('Train_reward', score, self.steps)
        self.trained_Q = self.Q
        
        
    def debug_train(self):
        """

        use the trained reward function to train the agent

        """
        state = self.env.reset()
        done  = False
        score = 0
        self.steps += 1
        while True:
            action = self.act(state, self.episode, True)
            next_state, _, done, _ = self.env.step(action)
            reward = self.r[state, action]
            self.optimize(state, action, reward, next_state, debug=True)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay* self.steps)
            score += reward
            if done:
                break
            state = next_state

        self.total_reward += score
        average_reward = self.total_reward / self.steps
        print("Total_steps {} Reward {:.2f} Average Reward {:.2f} epsilon {:.2f}".format(self.steps, score, average_reward, self.epsilon))

        
        
