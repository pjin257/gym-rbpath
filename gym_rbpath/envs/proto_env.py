import gym
from gym import spaces
import numpy as np
from scipy.special import comb
from sklearn.preprocessing import LabelBinarizer
import random
from gym_rbpath.envs.wheelnet import WheelNet
import logging

class RobustPathEnv(gym.Env):
    """A robust path routing environment for OpenAI gym"""

    def __init__(self):
        super(RobustPathEnv, self).__init__()

        # General variables defining the environment
        self.LINK_BW = 300
        self.NUM_NODES = 7
        self.NUM_CANDIDATES = 7
        self.wnet = WheelNet(num_of_nodes = self.NUM_NODES, weight = self.LINK_BW, num_of_candidates = self.NUM_CANDIDATES)
        self.NUM_LINKS = self.wnet.G.number_of_edges()
        self.NUM_ACTIONS = comb(self.NUM_CANDIDATES, 2)             # nCr = 21, where n = 7 and r = 2
        self.END_NODES = np.arange(0, self.NUM_NODES - 1, 1)        # [0, 1, 2, 3, 4, 5]
        self.DEMAND_BW = [1, 2, 8, 32, 64]
        self.ONEHOT_NODES = LabelBinarizer().fit_transform(self.END_NODES)  # one-hot encode the end nodes
        self.PENALTY_RATE = 0.5
        self.demand_src = 0
        self.demand_dst = 0
        self.demand_bw = 0

        # Action. Define what the agent can do
        # Primary and Alternative combination among "k" candidate paths 
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)

        # Observation
        low = np.zeros(79)
        high_tmp1 = np.append(np.full(12,1), np.asarray([64]))  # src, dst, bw
        high_tmp2 = np.append(np.full(self.NUM_LINKS, self.LINK_BW),     # current link state
                            np.full(self.NUM_CANDIDATES * (self.NUM_CANDIDATES - 1), self.NUM_NODES))    # number of shared NODES per path 
        high = np.append(high_tmp1, high_tmp2) 
        self.observation_space = spaces.Box(low, high, shape=
                        (79,), dtype=np.int32)
        
        # episode over
        self.episode_over = True
        self.info = {}

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []

        # draw wheel shaped network
        self.wnet.render()

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (That is, any links
                in the topology has been saturated)
            info (dict) :
                diagnostic information useful for debugging. It can sometimes
                be useful for learning (for example, it might contain the raw
                probabilities behind the environment's last state change).
                However, official evaluations of your agent are not allowed to
                use this for learning.
        """
        self.take_action(action)
        reward = self.get_reward(action) 
        ob = self.get_state() 

        return ob, reward, self.episode_over, self.info 

    def take_action(self, action):             
        # allocate link and check if any link is saturated
        if self.wnet.link_allock(self.demand_src, self.demand_dst, self.demand_bw, action):
            logging.info ('Link saturated, ending episode')
            self.episode_over = True

    def get_reward(self, act):
        # If any link is saturated and episoded ended, reward 0
        if self.episode_over:
            return 0

        # otherwise, reward for allocating path is the properly allocated bandwidth
        reward = self.demand_bw

        # penalty how many links are shared between primary and alternative paths
        selection = self.wnet.ACT_TABLE[self.demand_src][self.demand_dst][act]
        pri = set(selection[0])
        alt = set(selection[1])
        num_shared = len(pri & alt) - 2  # don't penalty on src/dst
        reward -= num_shared * self.PENALTY_RATE * self.demand_bw

        return reward

    def get_state(self):
        """Get the observation"""
        # sample src & dst pair
        sd_pair = random.sample(set(self.END_NODES), 2)
        self.demand_src = sd_pair[0]
        self.demand_dst = sd_pair[1]

        # onehot encode src & dst
        onehot_src = self.ONEHOT_NODES[sd_pair[0]]
        onehot_dst = self.ONEHOT_NODES[sd_pair[1]]
        onehot_sd_pair = np.append(onehot_src, onehot_dst)

        # sample demanded bandwidth and attach to demand
        self.demand_bw = random.choice(self.DEMAND_BW)
        demand = np.append(onehot_sd_pair, np.asarray([self.demand_bw]))

        # get available link capacity
        link_state = []
        for (u, v, wt) in self.wnet.G.edges.data('weight'):
            link_state.append(wt)

        # get the number of shared nodes between pri & alt paths
        ob = demand.tolist() + link_state + self.wnet.COM_LINK_TABLE[self.demand_src][self.demand_dst]
        
        return ob

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        # set available link capacity max
        for (u, v, wt) in self.wnet.G.edges.data('weight'):
            self.wnet.G[u][v]['weight'] = self.LINK_BW


        self.episode_over = False
        self.curr_episode += 1
        self.action_episode_memory.append([])
        return self.get_state()

    def render(self, mode='human', close=False):
        # no render. may be in future works?
        return 1   
