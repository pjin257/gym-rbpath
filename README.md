# gym-rbpath
 The Robust Path Routing environment is a single-agent domain featuring discrete observation and action spaces. In 14-node NSFNET topology, two routing scenarios below are surpported.
 
# k-shortest path scenario
The pool of candidate paths between every source-destination pair is pre-calculated by Dijkstra’s and Yen’s algorithms. Since Yen’s algorithm doesn’t guarantee disjoint paths, some of the calculated paths have shared nodes. In this scenario, the goal of the agent is 1) to serve as many as demands in limited total capacity and 2) to avoid selecting primary and protect paths pair that shares nodes as much as possible

# Disjoint path scenario
The pool of candidate paths is pre-calculated by the maximum-flow based disjoint path finding algorithm. Since disjoint paths do not share any node, the agent has one goal: to serve as many as demands in limited total capacity
