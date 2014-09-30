# Intro to AI - 440
# Kostas Bekris
## Fall 2014

---
---

## Table of Contents

1. [Problem-Solving Agents](#anchor1)
	2. [Search Problems](#anchor1.1)
	2. [Optimizing Search Problems](#anchor1.2)
	3. [Adversarial Search](#anchor1.3)


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

## [Optimizing Search Algorithms](id:anchor1.2)

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
	
## [Adversarial Search](id:anchor1.3)

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
		
---

## 9/16/14

## More Optimization

- $$$\alpha-\beta$$$ pruning depends on the order of terminal nodes
	- best-case: $$$O(b^{m/2})$$$
	- avg-case: $$$O(b^{3m/4})$$$
	- worst-case: $$$O(b^m)$$$
- Another improvement:
	- Cut the search early and apply heuristic evaluation function
- Heuristic on each node:
	- must order terminal nodes similary to the utility function
	- should be orders of magnitue faster than searching the subtree
	- for non-terminal nodes it should strongly correlate with the chances of winning from that node
- To optimize based on a heuristic, you must sample along a localized region around a maximum/minimum
	- **Nash Equilibrium**
		- Prisoner's Dilemma

---

## 9/18/14

### Local versus Global Search

#### Local Search

- Complete solution
- Searches how to change from invalid to valid solution

#### Global Search

- Incrememtally advances while maintaining validity from an empty to a complete solution


#### Hill Climbing problem

- Example of Local search

![Hill Climbing](http://trevoirwilliams.com/wp-content/uploads/2013/04/local-maximum0021-470x260.png)


### Probabilistic Completeness

- Complete algorithms will find a solution in finite time
- **Probabilistic** algorithms will find a solution eventually given infinite time
	- An algorithm is defined as **probabilistically complete** when the probability of finding a 
	- Given a algorithm with probability $$$p$$$ time of succeeding, the expected number of restarts is $$$1\over p$$$
	
### Local Beam Search

- Maintain a population of states
	- Apply "survival of the fittest"
	- We cannot blindly apply this rule, or we will eventually have multiple copies of the same state
	
### Genetic Algorithms

- Maintain a population of candidate solution states
- Apply a probabilistic selection step
	- High quality states have high prob. of surviving
- Then apply a crossover operation, which will combine  **string representations** of the states to guarantee new ones.
- Then apply a **mutation operation** over the string representation of the states.
- All these operations will be *randomly* chosen, unless we have some information about the state space
- This algorithm does not offer any *insight* into the search problems

### Constraint Safisfaction Problems

- **Constraint Satisfaction Problems** are used to conform a large, abstract set into a a standard, structured, and very simple representation
- Discrete and finite domains
- CSP has a set of variables, $$$X_1, X_2, $$$ and a set of constraints $$$C_1,\dotsc,C_m$$$
	- $$$X_i:\text{ has a domain }D_i$$$

#### Approach A: Incremental Formulation

- Initial assignment: empty
- Successor: pick an unassigned variable and assign it a value that does not violate constraints with the assigned variables
- Goal test: complete assignment
	- Backtracking Search (DFS)

---
## 9/25/14

### Searches with Logical Inference

#### Inference by Resolution

- Any complete search algorithm, applying only the resolution rule, can derive any conclusion entailed by a knowledge base
- Everything that can be expressed using propositional logic can use resolution to infer new knowledge

##### Steps

1. Represent your knowledge base as a set of logical propositions $$$(KB)$$$
2. Turn the statetment $$$(KB \land \neg\alpha)$$$ into **CNF** (conjunctive normal orm  form	
### Ground Resolution Thm

- If a set of clauses is unsatisfiable, then the resolution closure (the set of clauses you can generate) contains the empty clause
	- *Clause* - Huge boolean statement
	- Process will complete in finite time
- Not everything can be succinctly represented in propositional logic form

#### First-Order Logic

- Operators $$$\forall, \exists$$$

#### Second-order Logic

- Eventually
- Always
- (Modal Logic)

### Davis-Putnam Algorithm

- Backtracking search is still applicable to Boolean satisfiability
- Known as the **Davis-Putnam** algorithm
- Features of the algorithm:
	- **Early Termination**
		- No need to reach the leaf nodes to realize satisfiability or not
			- e.g. $$$(A\lor B)\land (A\lor C)$$$
			- True if $$$A$$$ is true, no need to search for $$$B$$$ or $$$C$$$
- A clause is true as long as one literal true
- A CNF sentence is false as long as one clause is false

#### Pure Symbol Heuristic

- A symbol is *pure* if it appears with the same "sign" in every clause
- Assign the value to the variable that satisfies the clauses

---
## 9/30/14

## Classical Planning

- Once you have expressed your planning problem using PDDL, you can then apply a search method
	- Forward Search
	- Consider all actions out of the state with pre-conditions attained by the state
	- You can also perform backward search where we make sure that the actions considered will have effects agreeing with the state
	
#### Challenges:

- Exploring irrelevant actions
- Huge state space

- We need heuristics for state abstraction, to group similar states to reduce the size of the search space
- Edges correspond to the PDDL actions to increase the number of edges we can relax the presentation of actions and compute the number of actions that can turn the current state into the good one

#### Branching factor of Backward Search

- Given a goal state $$$g$$$ and an action $$$a$$$, we can get the relevant state $$$g'$$$ $$g' = (g- ADD(a)\cup PRECOND(a))$$
- During backward search we need to deal with partially uninstantiated actions and states