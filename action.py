from nes_py.wrappers import JoypadSpace, FrameStack, GrayScaleObservation
from stable_baseline3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt 

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make("SuperMarioBros2-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

## shows the pixels in the game
# env.observation_space

## shows the possible actions to take
# env.action_space

done = True
for step in range(5000):
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()
