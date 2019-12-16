# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Script allowing to play the game by multiple players."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging


from gfootball.env import config
from gfootball.env import football_env
from gfootball.env import football_action_set
from gfootball.env import wrappers
import pgmmodel

import tensorflow as tf
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string('players', 'agent:left_players=1',
                    'Semicolon separated list of players, single keyboard '
                    'player on the left by default')
flags.DEFINE_string('level', 'academy_empty_goal_close', 'Level to play')
flags.DEFINE_enum('action_set', 'full', ['default', 'full'], 'Action set')
flags.DEFINE_bool('real_time', True,
                  'If true, environment will slow down so humans can play.')

def discount(rewards, discount_factor=.9):
    """
    Takes in a list of rewards for each timestep in an episode, 
    and returns a list of the sum of discounted rewards for
    each timestep. Refer to the slides to see how this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :return: discounted_rewards: list containing the sum of discounted rewards for each timestep in the original
    rewards list
    """
    # Compute discounted rewards

    steps = np.shape(rewards)[0]
    discounted_rewards = np.zeros(steps)
    
    discounted_rewards[steps-1] = rewards[steps-1]
    for i in range(steps-2,-1,-1):
        discounted_rewards[i] = discount_factor*discounted_rewards[i+1] + rewards[i]
    
    return discounted_rewards

def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :return: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps
    in the episode
    """
    simple115 = wrappers.Simple115StateWrapper(env)
    observations = []
    actions = []
    rewards = []
    observation = env.reset()
    ball_position = observation[0].get('ball')
    old_ball_x = ball_position[0]
    old_ball_y = ball_position[1]
    done = False
    while not done:
        
        # use model to generate probability distribution over next actions
        observation = simple115.observation(observation)
        observation = np.reshape(observation, (1, observation.shape[1]))
        observation = tf.convert_to_tensor(observation)
        observation = tf.dtypes.cast(observation, tf.float32)

        outputProbs = model.call(observation)
        amount_actions = np.shape(outputProbs)[1]

        # sample from this distribution to pick the next action
        action = np.random.choice(amount_actions, p=np.squeeze(outputProbs))
        observations.append(observation)
        actions.append(action)
        observation, reward, done, _ = env.step(action)
        ball_position = observation[0].get('ball')
        #print(ball_position)
        ball_x = ball_position[0]
        ball_y = ball_position[1]
        
        step_reward = -1
        if ((1-ball_x)**2+(0-ball_y)**2) < ((1-old_ball_x)**2+(0-old_ball_y)**2):
           step_reward = 1
        if reward == 1:
           step_reward = 300
           print("\nGOAL!!\n")
        rewards.append(step_reward)
        old_ball_x = ball_x
        old_ball_y = ball_y

    return observations, actions, rewards

def train(env, model):
  # Use generate trajectory to run an episode and get states, actions, and rewards.
    optimizer = model.optimizer
    
    with tf.GradientTape() as tape: 
        states, actions, rewards = generate_trajectory(env, model)
        # Compute discounted rewards.
        discount_rewards = discount(tf.convert_to_tensor(rewards))
        # Compute the loss from the model and run backpropagation on the model.
        loss = model.loss(tf.convert_to_tensor(states), tf.convert_to_tensor(actions), discount_rewards)
        print("Loss: ",loss.numpy())

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return np.sum(rewards)

def main(_):
  players = FLAGS.players.split(';') if FLAGS.players else ''
  cfg = config.Config({
      'action_set': 'full',
      'dump_full_episodes': True,
      'players': players,
      'real_time': False,
  })
  if FLAGS.level:
    cfg['level'] = FLAGS.level
  env = football_env.FootballEnv(cfg)
  #env.render()

  actions = football_action_set.get_action_set(cfg)

  state_size = 115
  num_actions = 9
  episodes = 1000
  model = pgmmodel.PGM(state_size, num_actions)

  try:

    for i in range(episodes):
        reward = train(env,model)
        print("Episode Reward: ", reward)

  except KeyboardInterrupt:
    logging.warning('Game stopped, writing dump...')
    env.write_dump('shutdown')
    exit(1)





if __name__ == '__main__':
  app.run(main)
