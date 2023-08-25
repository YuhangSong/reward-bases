# Basic Temporal Difference vs Reward Basis learners and Successor Representations
import numpy as np
from copy import deepcopy

class BaseLearner(object):
  def __init__(self, gamma, reward_function,env, learning_rate, beta):
    self.gamma = gamma
    self.reward_function = reward_function
    self.env = env
    self.lr = learning_rate
    self.beta = beta
    self.state = env.init()
    self.V = np.zeros(self.env.state_dim())
    self.action_space = self.env.action_space()
    self.homeostatic_agent = False
    
  def update_V(self, s,snext):
    pass

  def choose_action(self):
    pass

  def interact(self, N_episodes,return_actions = False, return_rb_vlist = False, return_mlist = False):
    total_rs = np.zeros(N_episodes)
    Vs = np.zeros((N_episodes, len(self.V)))
    if return_mlist:
      Ms = np.zeros((N_episodes,len(self.M), len(self.M)))
    if return_rb_vlist:
      Vss = np.zeros((N_episodes, len(self.alphas), len(self.V)))
    if return_actions:
      actions = []
    for n in range(N_episodes):
      #print(str(n) + " episodes")
      curr_s = self.env.reset()
      total_r = 0
      num_steps = 0
      while self.env.done is False and  num_steps <= 100:
        curr_s_idx = self.env.find_idx(curr_s)
        num_steps +=1
        #print("NUM STEPS: ", num_steps)
        if self.env.termination_condition(curr_s):
          r = self.update_V(curr_s, curr_s, terminal=True)
          total_r += r
        else:
          a = self.choose_action()
          if return_actions:
            actions.append(deepcopy(a))
          s = self.env.step(a)
          r = self.update_V(curr_s, s)
          total_r += r

        curr_s = deepcopy(s)

      total_rs[n] = total_r
      Vs[n, :] = deepcopy(self.V)
      if return_mlist:
        Ms[n,:,:] = deepcopy(self.M)
      if return_rb_vlist:
        Vss[n,:,:] = np.array(deepcopy(self.Vs))
    if return_actions:
      return total_rs, Vs, actions
    elif return_mlist:
      return total_rs, Vs, Ms
    elif return_rb_vlist:
      return total_rs, Vs, Vss
    else:
      return total_rs, Vs
    
    
class TD_Learner(BaseLearner):
  def __init__(self, gamma, reward_function,env, learning_rate, beta,random_policy = False):
    self.gamma = gamma
    self.reward_function = reward_function
    self.env = env
    self.lr = learning_rate
    self.beta = beta
    self.state = env.init()
    self.V = np.zeros(self.env.state_dim())
    self.action_space = self.env.action_space()
    self.homeostatic_agent = False
    self.random_policy = random_policy

  def update_V(self,s, snext, terminal=False):
    r = self.reward_function(self.env, s)
    s_idx = self.env.find_idx(s)
    snext_idx = self.env.find_idx(snext)
    if self.env.name != "salt_env":
      if not terminal:
        self.V[s_idx] = self.V[s_idx] + self.lr * (r + (self.gamma * self.V[snext_idx]) - self.V[s_idx])
      else:
        self.V[s_idx] = self.V[s_idx] + self.lr * (r - self.V[s_idx])
    else:
      if not terminal:
        self.V[s_idx] = self.V[s_idx] + self.lr * (r + - self.V[s_idx])
      else:
        self.V[s_idx] = self.V[s_idx] + self.lr * (r - self.V[s_idx])
    return r

  def softmax_choice(self, Qs):
    ps = np.exp(self.beta * Qs) / np.sum(np.exp(self.beta * Qs))
    a = np.random.choice(len(ps), p=ps)
    return a
  
  def compute_total_v(self):
    return self.V

  def choose_action(self):
    if self.random_policy:
      a = np.random.choice(len(self.action_space))
      return a
    else:
      Qs = np.zeros(len(self.action_space))
      # compute Q values from Vs by simulating updates
      for a in self.action_space:
        shat = self.env.step(a, simulated=True)
        s_idx = self.env.find_idx(shat)
        Qs[a] = self.V[s_idx]
      a = self.softmax_choice(Qs)
      return a


class Reward_Basis_Learner(TD_Learner):
  def __init__(self, gamma, rfuns,env, learning_rate, beta,alphas, random_policy = False):
    self.gamma = gamma
    self.rfuns = rfuns
    self.env = env
    self.lr = learning_rate
    self.beta = beta
    self.state = env.init()
    self.state_dim = env.state_dim()
    self.action_space = env.action_space()
    # reward function coefficient weights
    self.alphas = alphas
    assert len(self.alphas) == len(self.rfuns), "Number of reward functions and alphas must be the same"
    self.Vs = [np.zeros(self.state_dim) for i in range(len(self.alphas))]
    self.V = np.zeros(self.state_dim)
    self.random_policy = random_policy
    self.evaluation_reward_function = None

  def update_V(self,s, snext, terminal=False):
    rs = [r_fun(self.env,s) for r_fun in self.rfuns]
    s_idx = self.env.find_idx(s)
    snext_idx = self.env.find_idx(snext)
    for i in range(len(self.alphas)):
      if self.env.name == "salt_env":
        # single step env
        self.Vs[i][s_idx] = self.Vs[i][s_idx] + self.lr * (rs[i] - self.Vs[i][s_idx])
      else:
        if not terminal:
          self.Vs[i][s_idx] = self.Vs[i][s_idx] + self.lr * (rs[i] + (self.gamma * self.Vs[i][snext_idx]) - self.Vs[i][s_idx])
        else:
          self.Vs[i][s_idx] = self.Vs[i][s_idx] + self.lr * (rs[i] - self.Vs[i][s_idx])
    # compute total reward and value function
    if self.evaluation_reward_function is None:
      r = 0
      for i in range(len(self.alphas)):
        r += self.alphas[i] * rs[i]
    else:
      #print("USING EVALUATION REWARD FUNCTION")
      r = self.evaluation_reward_function(self.env, s)
      test_r = 0
      for i in range(len(self.alphas)):
        test_r += self.alphas[i] * rs[i]
    self.compute_total_v()
    return r

  def compute_total_v(self, alphas=None):
    if alphas is None:
      alphas = deepcopy(self.alphas)

    V = np.zeros_like(self.V)
    #print("ALPHAS: ", alphas)
    for i in range(len(alphas)):
      V += alphas[i] * self.Vs[i]
    self.V = deepcopy(V)
    return self.V

class Homeostatic_TD_Learner(TD_Learner):
  def __init__(self, gamma, reward_function,env, learning_rate, beta,kappa=None,simulated_reward_update=False):
    self.gamma = gamma
    self.reward_function = reward_function
    self.env = env
    self.lr = learning_rate
    self.beta = beta
    self.state = env.init()
    self.V = np.zeros(self.env.state_dim())
    self.action_space = self.env.action_space()
    self.simulated_reward_update = simulated_reward_update
    self.kappa = kappa
    self.homeostatic_agent = True
    if self.kappa is None:
      self.kappa = np.ones(len(self.V)) # homeostatic coefficient
      
  def set_kappa_reward_function(self):
    state_shape = self.env.state.shape
    kappa = np.zeros(np.prod(np.array(state_shape)))
    for idx in range(len(kappa)):
      pos = self.env.invert_idx(idx)
      r_predicted = self.reward_function(self.env, pos, simulated=True) 
      if r_predicted < 0:
        kappa[idx] = 1
      else:
        kappa[idx] = r_predicted
    self.kappa = deepcopy(kappa)
    return kappa

  def update_V(self,s, snext, terminal=False):
    r = self.reward_function(self.env, s)
    s_idx = self.env.find_idx(s)
    if not self.simulated_reward_update:
      r = self.kappa[s_idx] * r # homeostatic model
    snext_idx = self.env.find_idx(snext)
    if not terminal:
      self.V[s_idx] = self.V[s_idx] + self.lr * ((1 * r) + (self.gamma * self.V[snext_idx]) - self.V[s_idx])
    else:
      self.V[s_idx] = self.V[s_idx] + self.lr * ((1 * r)  - self.V[s_idx])
    return r

  def softmax_choice(self, Qs):
    ps = np.exp(self.beta * Qs) / np.sum(np.exp(self.beta * Qs))
    a = np.random.choice(len(ps), p=ps)
    return a

  def choose_action(self):
    Qs = np.zeros(len(self.action_space))
    # compute Q values from Vs by simulating updates
    for a in self.action_space:
      shat = self.env.step(a, simulated=True)
      s_idx = self.env.find_idx(shat)
      Qs[a] = self.V[s_idx]
      if self.simulated_reward_update:
        Qs[a] += self.kappa[s_idx] * self.reward_function(self.env, shat)
    a = self.softmax_choice(Qs - np.mean(Qs)) # baseline it so it doesn't explode 
    return a

class SuccessorRepresentationLearner(BaseLearner):
  def __init__(self, gamma, reward_function,env, learning_rate,beta, random_policy = False):
    self.gamma = gamma
    self.reward_function = reward_function
    self.env = env
    self.lr = learning_rate
    self.beta = beta
    self.state = env.init()
    self.action_space = self.env.action_space()
    self.state_dim = self.env.state_dim()
    print("STATE DIM: ", self.state_dim)
    #self.M = np.identity(self.state_dim)
    self.M = np.zeros((self.state_dim, self.state_dim))
    self.state_len = np.sqrt(self.state_dim)
    self.V = np.zeros(self.state_dim)
    self.random_policy = random_policy
  
  def idx_to_position(self, idx):
    x,y = divmod(idx, self.state_len)
    return np.array([int(x), int(y)])
    
  def position_to_idx(self, pos):
    return int((pos[0] * self.state_len) + pos[1])
    
  def get_rewards_all_states(self):
    rs = []
    for idx in range(self.state_dim):
      pos = self.idx_to_position(idx)
      #print("position: ", pos)
      r = self.reward_function(self.env, pos,simulated=True)
      rs.append(r)
    return np.array(rs)
  
  def compute_total_v(self):
    R = self.get_rewards_all_states()
    return np.dot(self.M, R)
  
  def compute_v_for_state(self, state):
    if not isinstance(state, int):
      state = self.position_to_idx(state)
    R = self.get_rewards_all_states()
    return np.dot(self.M[state, :], R)
  
  def onehot_state(self, state):
    if not isinstance(state, int):
      state = self.position_to_idx(state)
    z = np.zeros(self.state_dim)
    z[state] = 1
    return z
  
  def update_M(self, s1,s2):
    if not isinstance(s1, int):
      s1 = self.position_to_idx(s1)
    if not isinstance(s2, int):
      s2 = self.position_to_idx(s2)
    I = self.onehot_state(s1)
    self.M[s1,:] = self.M[s1,:] +  self.lr * (I + (self.gamma * self.M[s2,:]) - self.M[s1,:])
    
  def softmax_choice(self, Qs):
    ps = np.exp(self.beta * Qs) / np.sum(np.exp(self.beta * Qs))
    a = np.random.choice(len(ps), p=ps)
    return a

  def choose_action(self):
    if self.random_policy:
      a = np.random.choice(len(self.action_space))
      return a
    else:
      Qs = np.zeros(len(self.action_space))
      # compute Q values from Vs by simulating updates
      for a in self.action_space:
        shat = self.env.step(a, simulated=True)
        s_idx = self.env.find_idx(shat)
        Qs[a] = self.compute_v_for_state(s_idx)
      a = self.softmax_choice(Qs)
      return a
  
  def update_V(self, s, snext):
    r = self.reward_function(self.env, s,print_det=False)
    self.update_M(s, snext)
    self.V = self.compute_total_v()
    return r
    

    
    
  
  

          
  
  
  