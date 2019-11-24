# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0, 0)]
        for i in range(0, m):
            for j in range(0, m):
                if i != j:
                    self.action_space.append((i, j))
        self.state_space = [(i, j, k) for i in range(0, m) for j in range(0, t) for k in range(0, d)]
        self.state_init = self.state_space[np.random.choice(np.arange(0, len(self.state_space)))]

        # Start the first round
        self.reset()

    ## Encoding state (or state-action) for NN input

    # def state_encod_arch1(self, state):
    #    """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""

    #     return state_encod

    # Use this function if you are using architecture-2
    def state_encod_arch2(self, state, action):
        #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        state_encod = np.zeros(m + t + d + m + m)
        state_encod[m + int(state[1])] = state[1]
        state_encod[m + t + int(state[2])] = state[2]
        state_encod[m + t + d + int(action[0])] = action[0]
        state_encod[m + t + d + m + int(action[1])] = action[1]

        return state_encod

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        else:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15

        possible_actions_index = random.sample(range(1, (m - 1) * m + 1),
                                               requests)  # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        actions.append((0, 0))

        return possible_actions_index, actions

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        curr_state = int(state[0])
        start_state = int(action[0])
        end_state = int(action[1])
        time = int(state[1])
        day = int(state[2])
        if (action[0] == 0) & (action[1] == 0):
            reward = -C
        else:
            if (state[0] != action[0]):
                time_currLoc_to_start = Time_matrix[curr_state][start_state][time][day]
            else:
                time_currLoc_to_start = 0
            if time + time_currLoc_to_start > 23:
                time_state_to_end = Time_matrix[start_state][end_state][int(time + time_currLoc_to_start - 24)][day]
            else:
                time_state_to_end = Time_matrix[start_state][end_state][int(time + time_currLoc_to_start)][day]
            reward = (R * time_state_to_end) - (C * (time_currLoc_to_start + time_state_to_end))
        return reward

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        curr_state = int(state[0])
        start_state = int(action[0])
        end_state = int(action[1])
        time = int(state[1])
        day = int(state[2])
        if (start_state == 0) & (end_state == 0):
            next_state = (curr_state, Time_matrix[curr_state][curr_state][time][day], day)
        else:
            if (state[0] != action[0]):
                time_currLoc_to_start = Time_matrix[curr_state][start_state][time][day]
            else:
                time_currLoc_to_start = 0
            if time + time_currLoc_to_start > 23:
                time_state_to_end = Time_matrix[start_state][end_state][int(time + time_currLoc_to_start - 24)][day]
            else:
                time_state_to_end = Time_matrix[start_state][end_state][int(time + time_currLoc_to_start)][day]
            next_state = (end_state, time_state_to_end, state[2])

        return next_state

    def reset(self):
        return self.action_space, self.state_space, self.state_init