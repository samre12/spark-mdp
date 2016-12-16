# SPARK_MDP
This repository is for open source implementation of policy evaluation, policy iteration and value iteration for a given MDP using Apache-Spark as the computation framework.
The information about the state space, actions, transition probabilities and rewards should be supplied to the application using files stored on HDFS in the following format : 
_________________________Transition Probabilites___________________________:
a) Each line of input must correspond to a state of MDP in the order of the index of the state
b) Each line corresponding to state s must start with the index of the state s followed by delimiter "#" and then content in point (c)
c) Each line corresponding to state s must contain (action a, state s', probability p) triplets separated by "#" so that P(s, a, s') = p
   Example : 3#5,2,0.5#5,4,0.5#6,1,0.4#6,8,0.2#6,2,0.4
   If this is the input, this would mean that state number 3 can go to state number 2 with probability 0.5 and state number 4 with probability 0.5 upon taking action number 5 and similarly for action number 6. Thus, the sum of probabilities for each action must be 1 and should be checked upon by the user otherwise the application will raise an InvalidInputException
d) Entries not provided will be considered 0 by default.

 ______________________________Rewards_____________________________________:
a) Each line of the input must correspond to a (state, action) pair of the MDP
b) Each line must corresponding to state s and action a must contain the indices of the state s and the action a along with the reward of taking action a in state s
   Example : 3,5,4.3
   If this is the input, it corresponds to the fact that a reward of 4.3 units is received on taking action number 5 in state number 3
c) Entries not provided will be considered 0 by default.



