import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='disjoint-v0',
    entry_point='gym_rbpath.envs:DisjointPathEnv',
)

register(
    id='shortest-v0',
    entry_point='gym_rbpath.envs:ShortestPathEnv',
)

register(
    id='proto-v0',
    entry_point='gym_rbpath.envs:RobustPathEnv',
)