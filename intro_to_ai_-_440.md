# Intro to AI - 440
# Kostas Bekris
## Fall 2014

---
---

## Table of Contents

1. [Problem-Solving Agents](#anchor1)


---
---

## 9/4/14

# [Problem-Solving Agents](id:anchor1)

- Performance measure
	- Simplified
	- Corresponds to whether goal is reached

### Search Problems

- Need to define a state space $$$X$$$
- A successor function $$$X\times U\to X$$$
- Initial and goal state, and performance measure
- Applicable to pathfinding algorithms
- Need to avoid repeated states
- We need two data structures, one to con

#### Algorithm 1 (Tree Search)
	
	fringe = make.node(initial.node)
	loop
		if empty(fringe) then
			report failure
		node = get.node(fringe)
		if is.goal(node) then
			return node
		children = expand(node)
		fringe = add(children)
		
#### Algorithm 2 (Graph Search)

	closed = empty set
	fringe = make.node(initial.node)
	loop
		if empty(fringe) then
			report failure
		node = get.node(fringe)
		if is.goal(node) then
			return node
		if node not in closed then
			add node in closed
			children = expand(node)
			fringe = add(children)
			
Difference b/w Tree and Graph Search:

- Graphs cannot have loops
			
#### Comparing Search Strategies

1. Completeness
2. Optimality
3. Time Complexity: # of nodes generated during search
4. Space Complexity: maximum # of nodes stored in memory during search

##### Parameters

1. **b**: branch factor - # of successors per node
2. **d**: depth - level of shallowest goal node
3. **m**: max length: maximum length of any pathway

### Types of Searches

#### Breadth-First Search

- Fringe is stored as FIFO queue
- **Algorithm Evaluation:**
	- Complete
	- Optimal for single-source shortest path for unweighted graphs
		- As long as all the edges of a level have the same value
	- Time Complexity: $$\sum\_{i=0}^d b^d : O\big(b^d\big)$$
	- Space Complexity: $$\sum\_{i=0}^d b^d\text{ (closed list)} + b^{d+1}\text{ (fringe)} : O\big(b^{d+1}\big)$$
	
#### Depth-First Search

- Fringe stored as LIFO stack
- **Algorithm Evaluation**
	- Complete
	- Optimal??
	- Time Complexity: $$$O\left(b^m\\right)$$$
	- Space Complexity: $$$O\left(bm\right)$$$
- Time complexity is highly variable

#### Uniform-cost Search (Djikstra's)

- Fringe stored as a priority queue
- Priorities stored 

All these searches are called **uninformed searches**