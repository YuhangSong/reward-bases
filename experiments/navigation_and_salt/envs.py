# Environments for the experiments

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math

# Base Env class for subclassing to define the environment interface
class BaseEnv(object):
  def __init__(self):
    pass

  def init(self):
    pass

  def reset(self):
    pass

  def step(self):
    pass

  def state_dim(self):
    pass

  def find_idx(self):
    pass

  def termination_condition(self):
    pass

  def action_space(self):
    pass

class SeaSaltExperiment(BaseEnv):
  def __init__(self, step_threshold=10):
    self.state = np.zeros(2)
    # random initial start with salt
    self.state[0] = 1
    self.done = False
    self.steps = 0
    self.step_threshold = step_threshold
    self.name = "salt_env"

  def init(self):
    self.done = False
    return self.state

  def reset(self):
    self.state = np.zeros(2)
    self.state[0] = 1
    self.steps = 0
    self.done = False
    return self.state

  def state_dim(self):
    return len(self.state)

  def action_space(self):
    return [0,1]

  def find_idx(self,state):
    if state[0] == 1:
      return 0
    elif state[1] == 1:
      return 1
    else:
      raise ValueError("Invalid state")

  def step(self, a,simulated=False):
    state = np.zeros(2)
    state[a] = 1
    if not simulated:
      self.state = deepcopy(state)
      self.steps +=1
    return state

  def termination_condition(self,s):
    if self.steps >= self.step_threshold:
      self.done = True
    return self.done


# Two step task environment for Experiment 2
class TwoStepTaskEnv(BaseEnv):
  def __init__(self):
    self.state = np.zeros(5)
    self.state[0] = 1
    # setup transition matrices
    self.transition_matrix_left = np.zeros((5,5))
    self.transition_matrix_left[0,1] = 1
    self.transition_matrix_left[1,2] = 1
    self.transition_matrix_left[2,2] = 1
    self.transition_matrix_left[3,1] = 1
    self.transition_matrix_left[4,3] = 1
    #print(self.transition_matrix_left)
    self.transition_matrix_right = np.zeros((5,5))
    self.transition_matrix_right[0,3] = 1
    self.transition_matrix_right[1,3] = 1
    self.transition_matrix_right[2,1] = 1
    self.transition_matrix_right[3,4] = 1
    self.transition_matrix_right[4,4] = 1
    #print(self.transition_matrix_right)
    self.T_matrix = [self.transition_matrix_left.T, self.transition_matrix_right.T]
    self.done = False
    self.name = "two_step"

  def init(self):
    return self.state

  def reset(self):
    self.state = np.zeros(5)
    self.state[0] = 1
    self.done = False
    return self.state

  def state_dim(self):
    return len(self.state)

  def find_idx(self,vec):
    for (i, el) in enumerate(vec):
      if el == 1:
        return i

  def step(self, a,simulated=False):
    state = np.dot(self.T_matrix[a], self.state)
    if not simulated:
      self.state = deepcopy(state)
    return state

  def termination_condition(self, state):
    s_idx = self.find_idx(state)
    if s_idx == 2 or s_idx == 4:
        #print("CONDITION TRIGGERED")
        self.done = True
        return True

  def action_space(self):
    return [0,1]

class RoomEnv(BaseEnv):
  def __init__(self, room_size=6, p1=[1,2], p2=[4,4], p3 = [5,1], start_position=None, random_goal_positions = False):
    self.room_size = room_size
    if random_goal_positions:
      p1, p2, p3 = self.create_random_goal_positions()
      print("p1: ", p1)
      print("p2: ", p2)
      print("p3: ", p3)
    self.p1 = p1
    self.p2 = p2
    self.p3 = p2
    self.state = np.zeros((room_size, room_size))
    self.state[p1[0], p1[1]] = 2
    self.state[p2[0], p2[1]] = 3
    self.state[p3[0], p3[1]] = 4
    self.done = False
    self.position = self.get_start_position(start_position)
    self.p1_touched = False
    self.p2_touched = False
    self.p3_touched = False
    self.p1_rewarded = False
    self.p2_rewarded = False
    self.p3_rewarded = False
    self.name = "room"
    
  def generate_random_goal_positions(self):
    p1x = int(np.random.uniform(0, self.room_size))
    p2x = int(np.random.uniform(0, self.room_size))
    p3x = int(np.random.uniform(0, self.room_size))
    p1y = int(np.random.uniform(0, self.room_size))
    p2y = int(np.random.uniform(0, self.room_size))
    p3y = int(np.random.uniform(0, self.room_size))
    return p1x, p2x, p3x, p1y, p2y, p3y
  
  def verify_random_goal_positions(self,p1x, p1y, p2x, p2y, p3x, p3y):
    if (p1x - p2x)**2 <= 1 or (p1x - p3x)**2 <= 1 or (p1y - p2y)**2 <=1 or (p1y - p3y)**2 <=1:
      return False 
    if (p2x - p1x)**2 <= 1 or (p2x - p3x)**2 <= 1 or (p2y - p1y)**2 <=1 or (p2y - p3y)**2 <=1:
        return False 
    return True
  
  def create_random_goal_positions(self):
    p1x, p2x, p3x, p1y, p2y, p3y = self.generate_random_goal_positions()
    while not self.verify_random_goal_positions(p1x, p2x, p3x, p1y, p2y,p3y):
      p1x, p2x, p3x, p1y, p2y, p3y = self.generate_random_goal_positions()
    p1 = [p1x, p1y]
    p2 = [p2x, p2y]
    p3 = [p3x, p3y]
    return p1, p2, p3
  
  def visualize_env(self):
    plt.imshow(self.state)
    plt.xticks([])
    plt.yticks([])
    plt.show()  


  def get_start_position(self, start_position):
    if start_position is None:
      self.position = [np.random.randint(0,self.room_size), np.random.randint(0, self.room_size)]
      while self.state[self.position[0], self.position[1]] != 0:
        self.position = [np.random.randint(0,self.room_size), np.random.randint(0, self.room_size)]
    else:
      self.position = start_position
    return self.position

  def init(self):
    return self.position

  def state_dim(self):
    return self.room_size**2

  def check_points(self, pos):
    x,y = pos
    if self.state[x,y] ==2:
      self.p1_touched = True
    if self.state[x,y] == 3:
      self.p2_touched = True
    if self.state[x,y] == 4:
      self.p3_touched = True

  def reset(self,start_position=None):
    #print("ENV RESETTING")
    self.position = self.get_start_position(start_position)
    self.done = False
    self.p1_touched = False
    self.p2_touched = False
    self.p3_touched = False
    self.p1_rewarded = False
    self.p2_rewarded = False
    self.p3_rewarded = False
    return self.position

  def step(self, a, simulated=False):
    xpos,ypos = self.position
    if a !=0 and a != 1 and a != 2 and a != 3:
      raise ValueError("a must be 0,1,2,3") # catch if there is a bug here
    if a == 0:
      if xpos <= 0:
        xpos = 0
      else:
        xpos -= 1
    if a == 1:
      if xpos >= self.room_size-1:
        xpos = self.room_size -1
      else:
        xpos += 1

    if a == 2:
      if ypos <= 0:
        ypos = 0
      else:
        ypos -=1
    if a ==3:
      if ypos >= self.room_size - 1:
        ypos = self.room_size -1
      else:
        ypos += 1
    position = [xpos, ypos]
    if self.state[xpos, ypos] != 0:
      self.check_points(position) # note the flags
    if not simulated:
      self.position = deepcopy(position)
    return position

  def find_idx(self, pos):
    x,y = pos
    return (x * self.room_size) + y
  
  def invert_idx(self, idx):
    x,y = divmod(idx,  self.room_size)
    pos = [x,y]
    return pos

  def termination_condition(self, pos):
    # i.e. bumped into one of the endpoints
    if self.state[pos[0], pos[1]] != 0:
      self.done = True
      return True

  def all_points_termination_condition(self, pos):
    if self.p1_touched and self.p2_touched and self.p3_touched:
      self.done = True
      return True

  def action_space(self):
    return [0,1,2,3]

if __name__ == '__main__':
  print("RUNNING")
  room_env = RoomEnv(room_size=10, random_goal_positions = True)
  room_env.visualize_env()