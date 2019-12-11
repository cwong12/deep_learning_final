import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class PGM(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(PGM, self).__init__()
        self.num_actions = num_actions 

        # Define actor network parameters, critic network parameters, and optimizer

        self.hidden_size = 30

        self.actor1 = tf.keras.layers.Dense(self.hidden_size)
        self.actor2 = tf.keras.layers.Dense(self.num_actions)
        
        self.critic1 = tf.keras.layers.Dense(self.hidden_size)
        self.critic2 = tf.keras.layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        
        return

    @tf.function
    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        of each state in the episode
        """
        actor1out = self.actor1(states)
        relout = tf.nn.relu(actor1out)
        actor2out = self.actor2(relout)
        softout = tf.nn.softmax(actor2out)
        
        return softout

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode
        :return: A [episode_length] matrix representing the value of each state
        """
        # implement this :D

        critic1out = self.critic1(states)
        relout = tf.nn.relu(critic1out)
        values = self.critic2(relout)
        
        return values

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the handout to see how this is done.

        Remember that the loss is similar to the loss as in reinforce.py, with one specific change.

        1) Instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. Here, advantage is defined as discounted_rewards - state_values, where state_values is calculated by the critic network.
        
        2) In your actor loss, you must set advantage to be tf.stop_gradient(discounted_rewards - state_values). You may need to cast your (discounted_rewards - state_values) to tf.float32. tf.stop_gradient is used here to stop the loss calculated on the actor network from propagating back to the critic network.
        
        3) To calculate the loss for your critic network. Do this by calling the value_function on the states and then taking the sum of the squared advantage.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # implement this :)

        # Get the probs and values
        states = tf.reshape(states, [states.shape[0], states.shape[2]])
        probs = self.call(states)
        values = self.value_function(states)

        # Set up the indices matrix for tf.gather_nd
        indices = np.zeros([np.shape(states)[0], 2])
        for i in range(np.shape(states)[0]):
            indices[i,0] = i
            indices[i,1] = actions[i]  
        indices = tf.convert_to_tensor(indices)
        indices = tf.dtypes.cast(indices, tf.int32)

        # Get Paj
        Paj = tf.gather_nd(probs, indices)

        # Calculate advantage, actor loss, and critic loss
        advantage = tf.stop_gradient(tf.cast((discounted_rewards-values),tf.float32))
        loss_actor = tf.math.reduce_sum(tf.math.multiply(-tf.math.log(Paj),advantage))

        loss_critic = tf.math.reduce_sum(   tf.math.square(tf.cast((discounted_rewards-values),tf.float32))    )
        actor_weight=1
        critic_weight=1
        
        return actor_weight*loss_actor + critic_weight*loss_critic
