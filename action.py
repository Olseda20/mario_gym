# framestacking allows us to see the direction mario was moving in and such
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation

# when we implent the RL model we need to vectorize the model to order use it?
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from train_and_logging_callback import TrainAndLoggingCallback

from matplotlib import pyplot as plt

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import os

# 1- create the base environment
env = gym_super_mario_bros.make("SuperMarioBros2-v0")

# 2- simplify the environment
env = JoypadSpace(env, SIMPLE_MOVEMENT)
state = env.reset()
print("original shape", state.shape)

# 3- applying grayscale
env = GrayScaleObservation(env, keep_dim=True)
state = env.reset()
print("greyscale shape", state.shape)

# 4- wrap into dummy environment
env = DummyVecEnv([lambda: env])
state = env.reset()
print("vecotrised image shape", state.shape)

env = VecFrameStack(env, 4, channels_order="last")
state = env.reset()
print("vecotrised stack image shape", state.shape)

env.reset()
## shows the pixels in the game
# env.observation_space

## shows the possible actions to take
# env.action_space

# done = True
# for step in range(5000):
#     if done:
#         env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#     env.render()
#

plt.imshow(state[0])
plt.show()
env.close()
