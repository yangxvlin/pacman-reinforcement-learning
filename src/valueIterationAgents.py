# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

import numpy as np

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in np.arange(0, self.iterations):
            next_values = util.Counter()

            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue

                q_values = util.Counter()

                for action in self.mdp.getPossibleActions(state):
                    q_values[action] = self.computeQValueFromValues(state, action)

                key_max_value = q_values.argMax()
                next_values[state] = q_values[key_max_value]

            self.values = next_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        next_states_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0

        for (next_state, next_state_prob) in next_states_probs:
            q_value += next_state_prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)
        values = util.Counter()

        for action in actions:
            values[action] = self.computeQValueFromValues(state, action)

        policy = values.argMax()
        return policy


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for k in range(self.iterations):
            state = states[k % len(states)]

            if self.mdp.isTerminal(state):
                continue

            possible_actions = self.mdp.getPossibleActions(state)
            if not possible_actions:
                continue
            else:
                q_values = util.Counter()

                for action in self.mdp.getPossibleActions(state):
                    q_values[action] = self.computeQValueFromValues(state, action)

                key_max_value = q_values.argMax()
                self.values[state] = q_values[key_max_value]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Initialize an empty priority queue.
        pq = util.PriorityQueue()

        all_states = self.mdp.getStates()

        # Compute predecessors of all states.
        pred = {}
        for s in all_states:
            if self.mdp.isTerminal(s):
                continue
            for ac in self.mdp.getPossibleActions(s):
                for stt, _ in self.mdp.getTransitionStatesAndProbs(s, ac):
                    if stt in pred:
                        pred[stt].add(s)
                    else:
                        pred[stt] = {s}

        # For each non-terminal state s do:
        for s in all_states:
            if self.mdp.isTerminal(s):
                continue
            # Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s
            # (this represents what the value should be);
            # call this number diff
            diff = abs(self.values[s] - max([self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s)]))
            # Push s into the priority queue with priority -diff (note that this is negative).
            # We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
            pq.update(s, -diff)

        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for i in range(0, self.iterations):
            # If the priority queue is empty, then terminate.
            if pq.isEmpty():
                break
            # Pop a state s off the priority queue.
            s = pq.pop()
            # Update s's value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                self.values[s] = max([self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s)])

            # For each predecessor p of s, do:
            for p in pred[s]:
                # Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions
                # from p (this represents what the value should be);
                # call this number diff.
                # Do NOT update self.values[p] in this step.
                if self.mdp.isTerminal(p):
                    continue
                diff = abs(self.values[p] - max([self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)]))
                # If diff > theta, push p into the priority queue with priority -diff (note that this is negative), as long as it does not already exist in
                # the priority queue with equal or lower priority.
                # As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                if diff > self.theta:
                    pq.update(p, -diff)

