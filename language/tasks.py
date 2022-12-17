from random import randint
import pickle

class Task(object):
  """ Represents a task in the environment.
  
  """
  def __init__(self, env, start):
    self.env = env
    self.start = start
    # load saved task state
    with open('task_states/{}.pkl'.format(start), 'rb') as inp:
      newState = pickle.load(inp)
    env.restore_state(newState)
    env.step(0)


  def finished():
    # overridden by subclasses (specific tasks)
    # returns true if task is finished, false otherwise
    pass

  def reached_pos(self, x_, y_):
    # check if agent is at position (x_, y_)
    x, y = self.env.agent_pos()
    return (x_ - 5 <= x <= x_ + 5) and (y_ - 5 <= y <= y_ + 5)

  def reset(self):
    with open('task_states/{}.pkl'.format(self.start), 'rb') as inp:
      newState = pickle.load(inp)
    self.env.env.reset()
    self.env.env.restore_state(newState)
    obs, _, _, _ = self.env.step(0)
    return obs
    
class DownLadderJumpRight(Task):
  """
  Begins at game's start position. agent must climb down the ladder or jump right
  to reach the ladder on the lower right side of the room
  """
  def __init__(self, env):
    super().__init__(env, 'task_0')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.reached_pos(133, 192)

class ClimbDownRightLadder(Task):
  """
  Climb down the ladder on the right side of the start room
  """
  def __init__(self, env):
    super().__init__(env, 'task_1')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.reached_pos(133, 148)

class JumpSkullReachLadder(Task):
  """
  Jump over the skull and reach the ladder on the left side of the room
  """
  def __init__(self, env):
    super().__init__(env, 'task_2')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.reached_pos(20, 148)

class JumpSkullGetKey(Task):
  """
  More difficult version of JumpSkullReachLadder. Agent must jump over the skull,
  climb up the ladder and get the key
  """
  def __init__(self, env):
    super().__init__(env, 'task_2')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.env.has_key()


class ClimbLadderGetKey(Task):
  """ Climb up the ladder and get the key
  """
  def __init__(self, env):
    super().__init__(env, 'task_3')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.env.has_key()

class ClimbDownGoRightClimbUp(Task):
  """
  Climb down the ladder on the left, go right and climb up the ladder on the right
  """
  def __init__(self, env):
    super().__init__(env, 'task_4')
    self.env.repeat_action(0, 4)

  def finished(self):
    # top of the right ladder
    return self.reached_pos(133, 192)


class JumpMiddleClimbReachLeftDoor(Task):
  """Jump to the middle platform, climb up and reach the left door 
  """
  def __init__(self, env):
    super().__init__(env, 'task_5')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.env.room() == 4 and self.reached_pos(21, 235)