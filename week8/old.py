import numpy as np
from numpy.core.fromnumeric import argmax


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        # note seq length
        seq_length = len(seq)
        # hold the final states
        hidden_state_sequence = []

        # init DP tables to zeros of shape len of states X seq length
        table_1 = np.zeros(shape = (self.N, seq_length))
        table_2 = np.zeros(shape = (self.N, seq_length))
        
        # compute probabilities from all states to first observation
        for i in range(self.N): table_1[i, 0] = self.pi[i] * self.B[i][self.emissions.index(seq[0])]
        
        # Dp approach for all values , forward
        for j in range(1, seq_length):
            for i in range(self.N):
                
                temp_arr = [self.A[m][i] * self.B[i][self.emissions.index(seq[j])] * table_1[m, j-1] for m in range(self.N)]
                max_val = max(temp_arr)
                
                for t in range(len(temp_arr)):
                    if(temp_arr[t] == max_val):
                        max_index = t
                
                table_2[i, j] = max_index
                table_1[i, j] = table_1[max_index, j-1] * self.A[max_index][i] * self.B[i][self.emissions.index(seq[j])]

        temp_arr = [table_1[i, seq_length-1] for i in range(self.N)]
        max_val = max(temp_arr)
        for t in range(len(temp_arr)):
            if(temp_arr[t] == max_val):
                max_index = t
        
        # traverse backward to find the optimal sequence of states
        seq_length -= 1
        while(seq_length != -1):
            hidden_state_sequence.append(self.states[int(max_index)])
            max_index = table_2[int(max_index), int(seq_length)]
            seq_length -= 1
        
        return hidden_state_sequence