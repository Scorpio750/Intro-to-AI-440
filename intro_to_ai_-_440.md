# Intro to AI - 440
# Kostas Bekris
## Fall 2014

---
---

## Table of Contents

1. [Problem-Solving Agents](#anchor1)
	2. 


---
---

## 9/4/14

# [Problem-Solving Agents](id:anchor1)

- Performance measure
	- Simplified
	- Corresponds to whether goal is reached

## [Search Problems](id:anchor1.1)

- Need to define a state space $$$X$$$
- A successor function $$$X\times U\to X$$$
- Initial and goal state, and performance measure
- Applicable to pathfinding algorithms
- Need to avoid repeated states
- We need two data structures, one to con

### Algorithm 1 (Tree Search)
	
	fringe = make.node(initial.node)
	loop
		if empty(fringe) then
			report failure
		node = get.node(fringe)
		if is.goal(node) then
			return node
		children = expand(node)
		fringe = add(children)
		
### Algorithm 2 (Graph Search)

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
			
### Comparing Search Strategies

1. Completeness
2. Optimality
3. Time Complexity: # of nodes generated during search
4. Space Complexity: maximum # of nodes stored in memory during search

### Parameters

1. **b**: branch factor - # of successors per node
2. **d**: depth - level of shallowest goal node
3. **m**: max length: maximum length of any pathway

## Types of Searches

### Breadth-First Search

- Fringe is stored as FIFO queue
- **Algorithm Evaluation:**
	- Complete
	- Optimal for single-source shortest path for unweighted graphs
		- As long as all the edges of a level have the same value
	- Time Complexity: $$\sum\_{i=0}^d b^d : O\big(b^d\big)$$
	- Space Complexity: $$\sum\_{i=0}^d b^d\text{ (closed list)} + b^{d+1}\text{ (fringe)} : O\big(b^{d+1}\big)$$
	
### Depth-First Search

- Fringe stored as LIFO stack
- **Algorithm Evaluation**
	- Complete
	- Optimal??
	- Time Complexity: $$$O\left(b^m\\right)$$$
	- Space Complexity: $$$O\left(bm\right)$$$
- Time complexity is highly variable

### Uniform-cost Search (Djikstra's)

- Fringe stored as a priority queue
- Priorities stored 

All these searches are called **uninformed searches**

---

## 9/9/14

## Optimizing Search Algorithms

---

## 9/11/14

**Lemma:** if $$$h(n)$$$ is consistent then $$$f(n)$$$ is non-decreasing along any path.

\\[\begin{align}
g(n') &= g(n) + c(n,a,n) \\\
f(n') &= g(n') + h(n') \\\
h(n) &\leq c(n, a, n) + h(n') \\\
\implies h(n) &\geq h(n) - c(n, a, n') \\\
\end{align}\\]l

- The first goal node that we hit is going to be the optimal path
	- The heuristic at the goal node will always be $$$0$$$, and because **A\*** visits everything in non-decreasing order
- **A\*** expands all nodes with $$$f(n) < C^*$$$

### Heuristic Design

- In comparing heuristics we need to look at the *effective branching factor*
- If the algorithm has visited $$$N$$$ nodes and the solution depth was at $$$d$$$, then $$$b^*$$$ is the branching factor of the complete tree

- How do we come up with the heuristic value automatically?
	- We want to be able to quickly compute the costs
- What if you have multiple heuristics available?

\\[\begin{align}
&\text{for }\\{h_i,\dotsc, h_m\\} \\\
&h(n) = \max\\{h_i(n),\dotsc,h_m(n)\\} \\\
\end{align}\\]

- Solve a subproblem
	- improves solution times by a factor of 1000
	
## [Adversarial Search](id:anchor1.11)

- Games such as checkers, othello, chess, etc.
- They are hard search problems
	- Must predict possible search states
	
### Minimax Algorithm

- Player seeks to maximize algorithm, opponent seeks to minimize algorithm

		_Minimax_value(n): 
			if n is terminal:
				utility (n)
			if n is MAX state:
				max MV(s) for s in succ(n)
			if n is a MIN state:
				min MV(s) for s in succ(n)
				
- Efficiency:
	- $$$O(b^m)$$$ running time
	- $$$O(bm)$$$ space complexity

#### Speeding up the Minimax algorithm

- Prune the search tree
- Technique: **alpha-beta pruning**
	- Prune away nodes that cannot possibly influence the outcome
		- By this method we can escape searching subsets of the search tree
	- Performance depends on the order of terminal nodes instead
		- Best-case $$$O(b^{m / 2})$$$
		- Avg case: $$$O(b^{3m / 4})$$$