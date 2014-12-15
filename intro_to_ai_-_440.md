# Intro to AI - 440
# Kostas Bekris
## Fall 2014

---
---

## Table of Contents

1. [Problem-solving Agents](#anchor1)
	1. [Search Problems](#anchor1.1)
	2. [Optimizing Search Problems](#anchor1.2)
	3. [Adversarial Search](#anchor1.3)
	4. [Searches with Logical Inference](#anchor1.4)
		1. [Satisfiability](#anchor1.41)
	5. [Classical Planning](#anchor1.5)
	6. [Probabilistic Reasoning](#anchor1.6)
2. [Decision Theory](#anchor2)
	0. [Rules of Probability](#anchor2.01) 
	1. [Bayesian Networks](#anchor2.1)
	2. [Exact Inference using Bayesian Networks](#anchor2.2)
		1. [Inference by Enumeration](#anchor2.21)
		2. [Variable Elimination](#anchor2.22)
	3. [Approximate Inference](#anchor2.3)
		1. [Dynamic Bayesian Networks](#anchor2.31)
		2. [Filtering](#anchor2.32)
		3. [Continuous State Estimation](#anchor2.33)
	4. [Utility Theory](#anchor2.4)
		1. [Constraints on Rational Preferences](#anchor2.41)
	5. [Sequential Decision Making](#anchor2.5) 



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
\end{align}\\]

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
	- An algorithm is defined as **probabilistically complete** when the probability of finding a solution tends to 1, given infinite time.
		 $$\lim\_{t=\infty} P(a) = 1$$
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

## [Searches with Logical Inference](id:anchor1.4)

### [Satisfiability](id:anchor1.41)

- If a sentence $$$\alpha$$$ is true in **model** or **possible world** $$$m$$$, we say that *m* **satisfies** $$$\alpha$$$, or $$$m$$$ is a **model of** $$$\alpha$$$
- $$$M(\alpha)$$$ is the set of all possible worlds/models of $$$\alpha$$$

#### Entailment
\\[\begin{align}
\alpha \models \beta &\iff M(\alpha) \subseteq M(\beta) \\\
\alpha \equiv \beta &\iff \alpha \models \beta \wedge \beta \models \alpha \\\
\alpha \models \beta &\iff (\alpha \Rightarrow \beta) \\\
\end{align}\\]


### Inference by Resolution

- Any complete search algorithm, applying only the resolution rule, can derive any conclusion entailed by a knowledge base
- Everything that can be expressed using propositional logic can use resolution to infer new knowledge

#### Steps

1. Represent your knowledge base as a set of logical propositions $$$(KB)$$$
2. Turn the statement $$$(KB \land \neg\alpha)$$$ into **CNF** (conjunctive normal  form)

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

## [Classical Planning](id:anchor1.5)

- Expressed in PDDL (Planning Domain Definition Language)
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

---
## 10/2/14

## [Probabilistic Reasoning](id:anchor1.6)

- $$$A^*$$$ is optimally efficient; i.e. it expands the least amount of nodes
- Search tree differs in cases with multiple agents (chess)
	- If it is a zero-sum game, use mini-max algorithm
- What kind of heuristics are assigned to constraint satisfaction problems? 
	- Maximal constraint
	- If a knowledge base is expressed as a logical expression and you want to find if it expresses a value $$$a$$$, you use resolution:
	
	\\[\begin{aligned}
	&KB \models a? \\\
	(&KB ^ \neg a) \\\
	\Rightarrow &3-CNF \\\
	\because &\bot,\; KB \models a \\\
	\because \neg &\bot,\; KB \;\neg \models a
	\end{aligned}\\]
	
---

# [Decision Theory](id:anchor2)

- **Probability theory** + **Utility Theory**
- So far everything was either true or false
- But now we do not have enough information about the state of all variables
- In probability theory you assign a **belief** regarding the value of variables which can vary

### Decision-Theoretic Agents

- Update their belief about the state they are in based on actions/sensing
Calculate outcome probabilities for actions given their descriptions and current belief
- Select action that maximizes their expected utility

### [Rules of Probability](id:anchor2.01)

Probabilities are **stochastic**:

$$P(a) + P(\neg a) = 1$$

For state space $$$S = \\{s_1, s_2,\dotsc,s_n\\}$$$:

$$P(S) = \sum_i P(s_i) = 1$$


#### Conditional and Unconditional Probabilities


$$P(A\mid B) = {P(AB) \over P(B)}$$

#### Marginalization

$$P(Y) = \sum_Z P(YZ)$$

#### Conditioning rule

$$P(Y) = \sum_Z P(Y\mid Z) P(Z)$$

#### Normalization Factor

\\[\begin{align}
&P(A\mid B) + P(\neg A\mid B) = 1 \\\
&P(A\mid B) = {P(AB)\over P(B)} = \alpha P(AB) \\\
&P(\neg A\mid B) = {P\neg AB) \over P(B)} = \alpha P(\neg AB) \\\
\implies &\alpha (P(AB) + P(\neg AB) = 1 \\\
\implies &\alpha = {1\over P(AB) + P(\neg AB)} \\\
\end{align}\\

---
## 10/7/14

## More Decision Theory

- Recap: Maximize expected utility given uncertainty
- If $$$X, Y$$$ are independent events:

\\[\begin{aligned}
&P(X\mid Y) = P(X) \\\
&P(X, Y) = P(X) P(Y) \\\
&P(cause\mid effects_1,\dotsc,effects\_{10}) \\\
\end{aligned}\\]

### Baye's Rule

$$P(H\mid D) = {P(D\mid H)\; P(H)\over P(D)}$$

## [Representational Tool: Bayesian Networks](id:anchor2.1)

- Graphical Model
- Nodes are the random variables of a problem
- An edge $$$X\to Y$$$ implies that $$$Y$$$ depends on $$$X$$$
- Each $$$X_i$$$ has a conditional prob. distribution: $$P(X_i\mid Parents(X_i))$$

### Burglary Example

\\[\begin{pmatrix}
P(B) & & P(E) \\\
& P(A\mid BE) \\\
P(J\mid A) & & P(M\mid A) \\\
\end{pmatrix}\\]

$$\Downarrow$$

\\[\begin{pmatrix}
0.001 & & 0.002 \\\
& P(A\mid BE) \\\
.9 & & .7 \\\
\end{pmatrix}\\]

- Using the **full joint probability table** (product of all conditional probabilities on the Bayesian Network) with $$$2^5$$$ cells, we can use **marginalization** to reduce the amount of cells we have to look at

\\[\begin{align}
&P(X_3,X_5) = \sum\_{X\neq X_3, X_5}P(X_1,\dotsc,X_5) \\\
&P(X_3\mid X_5) = \alpha P(X_3,X_5) \\\
&P(\neg X_3\mid X_5 = \alpha P(\neq X_3, X_5) \\\
&P(X_1,\dotsc, X_n) = P(X_1,\dotsc,X\_{n-1})\; P(X_1 \dotso X\_{n-1}) \\\
=\; &P(X_n\mid X_1,\dotsc, X_n-1) \; P(X\_{n-1}\mid X_1,\dotsc X\_{n-2}\; P(X_1\dotso X\_{n-2}) \\\
=\; &\prod\_{x_i} P(X_i \mid X_1 \dotso X\_{i-1}) \\\
= &\prod\_{X_i} P(X_i\mid Parents(X_i)) \\\
\end{align}\\]

## [Exact Inference using Bayesian Networks](id:anchor2.2)

- $$$X:$$$ denotes a query variable
- $$$e:$$$ observed over evidence variables $$$E$$$
- $$$Y:$$$ hidden variables
- Complete set of variables: $$$X \cup E \cup Y$$$
- We want to compute $$$P(X\mid e)$$$

### 1. [Inference by enumeration](id:anchor2.21)

\\[\begin{align}
P(B\mid j,m) &= \alpha\, P(B,j,m) \\\
&= \alpha \sum_A \sum_E P(A, E, B, j=t, m=t) \\\
&= \alpha \sum_A \sum_E P(B)\; P(E)\; P(A\mid B,E)\; P(J=t, m=t) \\\
&\implies P(B\mid J=t, m=t) \\\
&= \alpha \, P(B) \; \sum_E P(E) \sum_AP(A\mid B, E) \; P(j=t\mid A)\; P(M=t\mid A) \\\
\end{align}\\]

### 2. [Variable Elimination](id:anchor2.22)

- Note inference by enumeration contains wasteful computation
	- We reuse intermediate results

---

## 10/14/14

## Probabilistic Inference

- **Naive Approach:** Go through the F.J.P.D.
- **Bayesian Networks** can express the (conditional) independent properties of a probabilistic setup
- Given a Bayesian Network, the FJPD equals the product of all conditional probabilities on the network
	- Inference by enumeration
	- Variable elimination extends this approach and aims to minimize repetitive computations by strong intermediant results
		- General case: exponential time
		- For *polytrees:* linear time (!!!)
	
\\[\begin{align}
P(A, B, C, D, E) &= P(A) \; P(B\mid A) \; P(C\mid A) \,\, P(D\mid B, C) \; P(E\mid C) \\\
\\\
P(A\mid B, E) &= \alpha P(A, B, E) = a \sum_D \sum_C P(A, B, C, D, E) \\\
P(A) &= aP(A) \; P(B\mid A) \sum_C P(C\mid A) \,\, P(E\mid C) \sum_D P(D\mid B, C) \\\
\end{align}\\]

## [Approximate Inference](id:anchor2.3)

- For unconditional probabilities: use direct sampling
- Direct sampling approximates the FJPD

$$P(a, b, c, d, e) \cong {\mu(a, b, c, d, e) \over N}$$

- As $$$N\to\infty$$$ the ratio on the right converges to the true probability
- For conditional probabilities we can use 
	- **Rejection sampling:** ignore all samples that do not agree with the evidence vars
	- **Likelihood weighting:** Fix the evidence variables. Keep track of a weight for each sample, that expresses how likely this sample is.
	
### [Dynamic Bayesian Networks](id:anchor2.31)

- Denotes observable evidence $$$E\_{t-1}, E_t, E\_{t+1}$$$ and and hidden state variables $$$X\_{t-1}, X_t, X\_{t+1}$$$
- Each state is dependent on the previous one, i.e. $$X\_{t+1} = P(X\_{t+1} \mid X_t)$$

#### Assumptions

- Markov assumption
- The current state depends only on a finite history of previous states
- 1st order Markov finite history = last state

#### Stationary Process

- The process with which states change over time and evidence variables depend on state variables remains the same

#### Three inputs for DBN

- Prior probability: $$$P(X_0)$$$
- Transition model: $$$P(X\_{t+1} \mid X_t)$$$
- Observation model: $$$P(E\_{t+1}\mid X\_{t+1})$$$

The **FJPD** is:

\\[\begin{align}
&P(X_0,\dotsc, X_t, E_1,\dotsc, E_t) \\\
=\; &\text{Prior Probability} \bigg(\prod\_{k=1}^t \text{Transition model} * \text{Observation model}\bigg) \\\
=\; & P(X_0) \prod\_{k=1}^t P(X_k\mid X\_{k+1})\; P(E_k\mid X_k) \\\
\end{align}\\]

- Problems we want to answer in an incremental way:
	- Filtering problem $$$P(X_t\mid E\_{1:t})$$$
	- Prediction Problem $$$P(X\_{t+k}\mid E\_{1:t}) \quad k\geq 1$$$
	- Smoothing Problem $$$P(X_k \mid E\_{1:t}) \quad 1\leq k < t$$$
	
### [Filtering](id:anchor2.32)

- The idea is to solve this incrementally

$$P(X\_{t+1}\mid E\_{1:t})$$

- Assume that you have solved this problem in the previous time step

$$P(X_t \mid E\_{1:t})$$

\\[\begin{align}
&P(X\_{t+1} \mid E\_{1:t+1}) = P(X\_{t+1}\mid E\_{1:t} E\_{t+1}) \\\
\text{Apply Bayes's Rule} \\\
 \hookrightarrow &= \alpha P(E\_{t+1} \mid X\_{t+1} E\_{1:t} \; P(X\_{t+1} \mid E\_{1:t}) \\\
\text{Markov assumption} \\\
 \hookrightarrow &= \alpha P(E\_{t+1} \mid X\_{t+1} \; P(X\_{t+1} \mid E\_{1:t}) \\\
\end{align}\\]

---
## 10/16/14

## More Probabilistic Inference

- Transition Model
- Observation Model
- Input: $$$P(X_0), TM, OM$$$
- Problems: 
	- Filtering: $$$P(X_t\mid E\_{1:t})$$$
	- Prediction: $$$P(X\_{t+k} \mid E\_{1:t})\quad k\geq 1$$$
	- Smoothing: $$$P(X_k\mid E\_{1:t})\quad 1\leq k < t$$$
	- Most Likely Explanation: $$$\mathrm{argmax}\_{x\_{1:t}}P(X\_{1:t} \mid E\_{1:t})$$$

### Forward + Backward Recursion

- Constant Operator $$$\to$$$ each stop
- For $$$t$$$ states, $$$O(t)$$$ cost
- The same for backward
- We have to store intermediary results.
- Similar to filtering:

\\[\begin{align}
&\mathrm{argmax}\\{P(X\_{i:t+1} \mid E\_{i:t+1})\\} \\\
= &\alpha \, P(E\_{t+1}\mid X\_{t+1}) \; \mathrm{argmax}\\{P(X\_{t+1}\mid X_t)\\} \; \mathrm{argmax}\\{P(X\_{1:t})\mid E\_{1:t})\\} \\\
\end{align}\\]

### [Continuous State Estimation](id:anchor2.33)

- Impossible to specify conditional probabilities $$$\forall$$$ values
	1. Discretization is lossy
	2. Use **probability density function**
		
		$$$\hookrightarrow$$$ Should require finite # of params

**Ex:** Gaussian Distribution

$$\mathcal{N}(\mu,\sigma^2), \;\mu = \text{mean}, \sigma = \text{std. dev.}$$

---
## $$$\S$$$ [Utility Theory](id:anchor2.4)

- The probability of outcome $$$s'$$$ occurring, given evidence observations $$$\mathbf e$$$ and random variable $$$\mathrm{RESULT}(a) \ni \mathrm{RESULT}$$$ is a *random variable* with action $$$a$$$ is written as $$P(\mathrm{RESULT}(a) = s'\mid a,\mathbf e)$$
- To maximise utility, we use a **utility function** $$$U(s)$$$ which assigns a single number to express the desirability of a state.
- **Expected utility** is the sum of all the probabilities of the events in the state space occurring times their utility[^1]: $$EU(a \mid \mathbf e) = \sum\_{s'}P(\mathrm{RESULT}(a) = s' \mid a, \mathbf e) \; U(s')$$

[^1]: In this sense, we are building theoretic agents which are **act utilitarian**. What about building **deontological** theoretic agents? Would it be easier to implement supposing we loaded them with a set of *categorial imperatives*?

### Maximum Expected Utility

- The principle of **maximum expected utility** (MEU) says that a rational agent should choose the action that maximizes the agent's expected utility: $$action = \mathrm{argmax}\_a\;EU(a\mid \mathbf e)$$


### [Constraints on Rational Preferences](id:anchor2.41)

We use the folowing notation to describe an agent's preferences:

1. $$$A \succ B: \quad \text{the agent prefers} A \text{ over } B$$$
2. $$$A \sim B: \quad \text{the agent is indifferent b/w } A \text{ and } B$$$
3. $$$A \succsim B: \quad \text{the agent prefers } A \text{ to } B \text{ or is indifferent}$$$

- We can think of the set of possible outcomes for each action as a **lottery**. A lottery $$$L$$$ with possible outcomes $$$S_1,\dotsc,S_n$$$ that occur with probabilities $$$p_1,\dotsc,p_n$$$ is written as: $$L = \[p_1, S_1; p_2, S_2;\dotsc, p_n, S_n\]$$
- In general, each outcome $$$S_i$$$ of a lottery can be either an atomic state or another lottery.


We have six *constraints* that all reasonable preference relations must obey:

1. **Orderability:** Given any two lotteries, a rational agent must prefer prefer one to the other or else rate the two as equally preferable. The agent cannot avoid deciding. Refusing to bet is like refusing to allow time to pass. $$(A \succ B) \vee (B \succ A) \vee (A \sim B)$$
2. **Transitivity:** Given any three lotteries, if an agent prefers $$$A$$$ to $$$B$$$ and prefers $$$B$$$ to $$$C$$$, then the agent must prefer $$$A$$$ to $$$C$$$. $$(A \succ B) \wedge (B \succ C) \implies (A \succ C)$$
3. **Continuity:** If some lottery $$$B$$$ is between $$$A$$$ and $$$C$$$ in preference, then there is some probabilty $$$p$$$ for which the rational agent will be indifferent between getting $$$B$$$ for sure and the lottery that yields $$$A$$$ with probability $$$p$$$ and $$$C$$$ with probability $$$1-p$$$.[^2] $$A \succ B \succ V \implies \exists p \, \left\[ \, p, A; 1-p, C \, \right\] \sim B$$
4. **Substitutability:** If an agent is indifferent b/w two lotteries $$$A$$$ and $$$B$$$, then the agen is indifferent b/w two more complex lotteries that are the same except that $$$B$$$ replaces $$$A$$$ in one of them. This holds regardless of the probabilities and the other outcomes in the lotteries. This also holds if we substitute $$$\succ$$$ for $$$\sim~$$$. $$A \sim B \implies [\,p, A; 1-p, C\,] \sim [\,p, B; 1-p, c\,]$$
5. **Monotonicity:** If $$$A$$$ and $$$B$$$ have the same two possible outcomes $$$S_1$$$ and $$$S_2$$$. If an agent prefers $$$S_1$$$ to $$$S_2$$$ then the agent must prefer the lottery that has a higher probability for $$$S_1$$$ and vice versa: $$S_1 \sim S_2 \implies \left\\{\left(p > q \iff [\,p, S_1; 1-p, S_2\,] \succ [\,q, S_1; 1-q, S_2\,]\right)\right\\}$$
6. **Decomposability:** Compound lotteries can be reduced to simpler ones using the laws of probability. This is also called the "no fun in gambling" rule because it says that two consecutive lotteries can be compressed into a single equivalent lottery. $$\left[\,p,A; 1-p,\left[\,q, B; 1-q, C\,\right]\right] \sim \left[\, p, A; (1-p)q, B; (1-p)(1-q), C\,\right]$$

We can remember these constraints with the mnemonic **COM STD** (pronounced cum ess tee dee, thanks P.J. Tai)


[^2]: I believe this means that there was be some probability for which the weighted outcomes of $$$A$$$ and $$$C$$$ will the utility of $$$B$$$

### Expected Utility of a Lottery

- The utility of a lottery is the sum of the probability of each outcome times the utility of that outcome: $$U\left(\left[\,p_1, S_1;\dotsc;p_n,S_n\,\right]\right) = \sum_i p_i U(S_i)$$

- In a *deterministic environment* an agent only needs a preference ranking on states. This is called a **value function** or **ordinal utility function**.
- We typically fix the *best possible prize* at $$$U(S) = u\_\top$$$ and a *worst possible prize* at $$$U(S) = u\_\bot$$$
	- **Normalized** utilities use a scale with $$$u\_\bot = 0, u\_\top = 1$$$
- Agents that prefer a sure thing with a payoff less than the expected value are **risk-averse**. The opposite is **risk-seeking**.
- The value an agent will accept in lieu of a lottery is called the **certainty equivalent**
	- Studies show most people will accept about $400 in lieu of a gamble that gives a 50/50 shot at winning $1000
	
## [Decision Networks](id:anchor2.5)

- We can represent the total state space of an agent's possible decisions with a **decision network**
- Decision networks combine Bayesian netowkrs with additional node types for actions and utilities.
- There are 3 kinds of nodes for a DN:

1. Chance nodes $$$\bigcirc$$$ = random variables.
2. Decision nodes $$$\Box$$$ = actions.
3. Utility nodes (value nodes) $$$\diamond$$$ = the agent's utility function.

Decision networks are valuated according to the following algorithm:

1. Set evidence variables for current state.
2. For each possible vlue of that devision node:
	1. Set decision node to that value
	2. Calculate the posterior probabilities for the parent nodes of the utility node, using a standard probabilistic inference algorithm.
	Calculate the resulting utility for the action.
3. Return the action with the highest utility

---

## 10/30/14

## [Sequential Decision Making](id:anchor2.5)

- For imperfect actions, there is a probability that you actually follow the correct path.

### Components of such Problems

1. Transition model $$T(S,A, S')$$ For the time being assume we can perfectly sense our state (no need for observation model).
2. Reward for each state $$$s,\;R(s)$$$.
	- Think of it as an incentive to get to the goal problem


### Policies

A fixed action sequence may always not be a desirable solution

- Instead, a solution should specify what the agent should do at each state
	- These solutions are called **policies:** $$\Pi(s): \text{ action recommended by policy at state }s$$
	- **Ex:** treat a state space as a *vector field*
- Each time we execute a policy, a different state sequence arises
- We must look at the *expected utility* of our policies
- *Policies* correspond to a *distribution* of path sequences
	- There is probability $$$p$$$ we take path $$$\alpha$$$, prob. $$$q$$$ we take path $$$\beta$$$, etc.
- Eventually our goal is to select the policy $$$\pi^\*$$$ that optimizes *expected utility*
- If you have $$$\pi^*$$$ how should you move?
	- Execute $$$\pi^*(s)$$$ at $$$s$$$
	
#### Finding Utility

- What is the utility of a path sequence?
	- One idea: sum the rewards of individual states $$U = \sum_1^n r(s_i)$$
		- There is an issue with the horizon of the planning process
			- As infinite sequences may arise, the utility of such sequences may become infinitely large
			- If we have a stochastic model, we can do... something
\\[\begin{align}
U\left[(s_0, s_1,\dotsc)\right] = 0 \leq \gamma \leq 1 \\\
R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2)\dotsc \\\
\end{align}\\]
With discounted rewards,
\\[\begin{align}
U\left[(s_0, s_1, \dotsc)\right] &= \sum\_{t=0}^\infty \gamma^t R(s_t) \sum\_{t=0}^\infty \gamma^i R\_{max} \\\
&= {R\_{max} \over 1-\gamma} \\\
\end{align}\\]

When we want to find the optimum policy we want to maximize expected utility

$$\Pi^* = \mathrm{argmax} \; E\left\[ \sum\_{t=0}^\infty \gamma^t R(s_t) \mid \Pi\right\]$$

1. **Value Iteration**
	- If you calculate the utility of each state, you can then compute the optimum action at each state
	- **Utility of a state**: Expected utility of the state sequences that might arise given a policy. $$$U^\Pi (s)$$$
	
	$$U(s) = U^{\Pi^*} (s)$$
	$$U^\Pi (s) = E\left\[ \sum\_{t=0}^\infty \gamma^t R(s_t) \mid \Pi, s_0 = s\right\]$$
	
The above equation is one of the most important equations in AI, known as **Bellman equation**