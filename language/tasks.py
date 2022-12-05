from random import randint
import pickle

class Task(object):
  def __init__(self, env, start):
    self.env = env
    self.start = start
    with open('task_states/{}.pkl'.format(start), 'rb') as inp:
      newState = pickle.load(inp)
    env.restore_state(newState)
    env.step(0)
    # self.start = self.env.load_state('game_state_ckpts/{}.npy'.format(start))

  def finished():
    pass

  def reached_pos(self, x_, y_):
    x, y = self.env.agent_pos()
    return (x_ - 5 <= x <= x_ + 5) and (y_ - 5 <= y <= y_ + 5)

  def reset(self):
    with open('task_states/{}.pkl'.format(self.start), 'rb') as inp:
      newState = pickle.load(inp)
    self.env.env.reset()
    self.env.env.restore_state(newState)
    obs, _, _, _ = self.env.step(0)
    #print('reset at {}'.format(self.env.n_steps))
    #print('task reset: ', self)
    return obs
    # self.start = self.env.load_state('game_state_ckpts/{}.npy'.format(start))
    
class DownLadderJumpRight(Task):
  def __init__(self, env):
    super().__init__(env, 'task_0')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.reached_pos(133, 192)

class ClimbDownRightLadder(Task):
  def __init__(self, env):
    super().__init__(env, 'task_1')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.reached_pos(133, 148)

class JumpSkullReachLadder(Task):
  def __init__(self, env):
    super().__init__(env, 'task_2')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.reached_pos(20, 148)

class JumpSkullGetKey(Task):
  def __init__(self, env):
    super().__init__(env, 'task_2')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.env.has_key()


class ClimbLadderGetKey(Task):
  def __init__(self, env):
    super().__init__(env, 'task_3')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.env.has_key()

class ClimbDownGoRightClimbUp(Task):
  def __init__(self, env):
    super().__init__(env, 'task_4')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.reached_pos(133, 192)


class JumpMiddleClimbReachLeftDoor(Task):
  def __init__(self, env):
    super().__init__(env, 'task_5')
    self.env.repeat_action(0, 4)

  def finished(self):
    return self.env.room() == 4 and self.reached_pos(21, 235)


# class Task6(Task):
#   def __init__(self, env):
#     super(Task6, self).__init__(env, 'ckpt-6')
#     self.env.repeat_action(0, 4)

#   def finished(self):
#     return self.env.room() == 3 and self.reached_pos(77, 137)


# class Task7(Task):
#   def __init__(self, env):
#     super(Task7, self).__init__(env, 'ckpt-7-8')
#     self.env.repeat_action(0, 4)

#   def finished(self):
#     return self.env.room() == 10 and self.reached_pos(13, 235) and self.env.orb_collected()


# class Task8(Task):
#   def __init__(self, env):
#     super(Task8, self).__init__(env, 'ckpt-7-8')
#     self.env.repeat_action(0, 4)

#   def finished(self):
#     return self.env.room() == 10 and self.reached_pos(138, 235)


# class Task9(Task):
#   def __init__(self, env):
#     super(Task9, self).__init__(env, 'ckpt-9')
#     self.env.repeat_action(0, 4)

#   def finished(self):
#     return self.env.room() == 10 and self.reached_pos(138, 235)


# class Task10(Task):
#   def __init__(self, env):
#     super(Task10, self).__init__(env, 'ckpt-10')
#     self.env.repeat_action(0, 4)

#   def finished(self):
#     return self.env.room() == 9 and self.reached_pos(8, 235)


# class Task11(Task):
#   def __init__(self, env):
#     super(Task11, self).__init__(env, 'ckpt-11-12')
#     self.env.repeat_action(0, 4)

#   def finished(self):
#     return self.env.room() == 8 and self.reached_pos(77, 155)


# class Task12(Task):
#   def __init__(self, env):
#     super(Task12, self).__init__(env, 'ckpt-11-12')
#     self.env.repeat_action(0, 4)

#   def finished(self):
#     return self.env.room() == 8 and self.reached_pos(77, 235)


# class Task13(Task):
#   def __init__(self, env):
#     super(Task13, self).__init__(env, 'ckpt-13')
#     self.env.repeat_action(0, 4)

#   def finished(self):
#     return self.env.room() == 8 and self.reached_pos(77, 235) and self.env.has_key()


# class Task14(Task):
#   def __init__(self, env):
#     super(Task14, self).__init__(env, 'ckpt-14')
#     self.env.repeat_action(0, 4)

#   def finished(self):
#     return self.env.room() == 8 and self.reached_pos(77, 155) and self.env.has_key()


# class Task15(Task):
#   def __init__(self, env):
#     super(Task15, self).__init__(env, 'ckpt-15')
#     self.env.repeat_action(0, 4)

#   def finished(self):
#     return self.env.room() == 8 and self.reached_pos(149, 235)
