# SECTION 2 ARTICLE : _Practical bayesian optimization of machine learning algorithms_
> **Citation authors**: In this section we briefly review the general Bayesian
optimization approach, before discussing our novel contributions in Section 3

## 2) BAYESIAN OPTIMIZATION WITH GAUSSIAN PROCESS PRIORS

### TARGET ?
=> **finding the minimum of a function f(x)** amoung some **boundset X** (ensemble bornée X)
with X as **subset of R**.

### SPECIFICITY ?
=> **Generate a probabilistic model for f(x)** and use it to **decide amoung X** where to make **the next
evaluation** _(taking in consideration uncertainty)_.

### PRINCIPLE / IDEA ?
=> **Use all information given by the previous evaluation of f(x)** _(not only **local gradient** and 
**hessian approximations**)_.

### BENEFITS ?
=> Possibility of **finding the minimum of a complex non-convex functions** with only **few 
evaluations**.

### DRAWBACK ?
=> **Request** more **extra computation** to determine the next point to try. 
=> But **extra computation** is **not really a big deal when training a machine learning algo**.

### HOW IT WORKS ?
=> **2 MAJOR CHOICES to make**:
  - 1. **select amoung previous functions**, functions that will **give assumptions(more informations)
  about the function being optimized**. And for this, they choose **Gaussian Process Priors** that
  is more efficient performing such task.
  
  - 2. **choose an acquisition function to generate a utility function** from previous model, in order to 
  **find the next point to evaluate**.
  (Deuxièmement, nous devons choisir une fonction d'acquisition, qui est utilisée pour construire une fonction 
  d'utilité à partir du modèle postérieur, ce qui nous permet de déterminer le point suivant à évaluer.)


### 2.1 GAUSSIAN PROCESS

The **Gaussian Process** is a **convenient and powerful** prior distribution on functions which we
will take here to be the form _f : X -> R_. (explain how GP work [lien video youtube](https://www.youtube.com/watch?v=vU6AiEYED9E&t=539s)) ( Le processus gaussien (GP) est une distribution a priori pratique et puissante sur les fonctions, que nous prendrons ici pour être de la forme f: X → R).


### 2.2 ACQUISITION FUNCTIONS FOR BAYESIAN OPTIMIZATION

#### ROLE ? 
=> **Generate a utility function** from previous model **to find the next point to evaluate**.

#### HOW IT WORKS ?
1. We denote the acquisition function by : **a : X -> R+**
2. We suppose that **the function f(x)** is drawn from a **Gaussian process prior**. 
=> So as we seen previously, the results are of the form {X_1...X_n, Y_1...Y_n} that we will denote 
by **{Xn , Yn}** with Yn ~ N(f(Xn),v).
3. Several different function will be proposed.
4. In general these acquisition functions depend on **previous observations and GP hyperparameters** 
( we denote this dependence as **a(X; {Xn,Yn}, θ)** ).
5. These function depend only on the model through its **predictive mean function  μ (X; {Xn, Yn}, θ)** and 
**predictive variance function σ 2 (X ; {Xn , Yn}, θ)**.
6. We will denote the best current value as **Xbest = argmin_Xn f(Xn)**.
7. We will denote the **cumulative distribution function of the standard normal** as **Φ(·)**.
8. Determine what point in X should be evaluated next via a proxy optimization  **Xnext = argmax_x a(X)**.

#### 3 DIFFERENT STRATEGIES TO FIND ACQUISITION FUNCTIONS 

#### STRATEGY 1 : Probability of Improvement
##### PRINCIPLE ?
=>  **maximize the probability of improving over the best current value**
##### HOW ?
Under GP this can be computed analytically as : 
  (1) # polycop
  
#### STRATEGY 2 : Expected Improvement
##### PRINCIPLE ?
=> **maximize the expected probability of improving over the best current value**.
##### HOW ?
Under GP this can be computed analytically as : 
  (2) # polycop
  
#### STRATEGY 3 : GP Upper Confidence Bound
##### PRINCIPLE ?
=> (more recent one) **construct acquisition functions that minimize regret over the course of 
their optimization.**
##### HOW ?
Under GP this can be computed analytically as : 
  (3) # polycop
  
#### IN THIS WORK ?
=> **They use EI **
**WHY ?**
-----------
1. Better-behaved (compared to __probability of improvement__)
2. doesn't require its own tuning parameter (comparted to __GP upper conf__)
3. performs well in minimization problems.




# SECTION 3 ARTICLE : _Practical bayesian optimization of machine learning algorithms_

## LIMITATIONS
+ As we 've seen previously in section 2, Bayesian Optimization with GP is an elegant and effective 
framework for optimizing expensive functions , but have limitations when it comes to optimizing
hyperparameters in machine learning problems.

## WHY ?
- 1. Its unclear (difficult) how to choose the **convariance function and its associated hyperparameters**.
- 2. As the function evaluation itself may involve a time-consuming optimization procedure. Durations of problems 
can vary and must be taken in account.
- 3. It doesn't take advantage of multi-core parallelism to adapt well in modern computationnal environments.

## WHAT ?
> **Citation authors** : In this section, we propose solutions to each of these issues.

## 3 PRACTICAL CONSIDERATIONS FOR BAYESIAN OPTIMIZATION OF HYPERPARAMETERS

### 3.1 Covariance Functions and Treatment of Covariance Hyperparameters ( Solve LIMITATION 1)

#### PROBLEM ?
=> The power of GP to express a rich distribution on functions depends essentially on the covariance functions.
  And in particular the **Automatic Relevance Determination (ARD)** _squared exponential kernel_ we denote by K_SE(X,X')
  (4) # polycop is often a default choice for GP regression but sample function that we obtain from this covariance function are
  unrealistic for practical optimization problem.
  
#### SOLUTION ?

##### Choose the form of the covariance function :
=> They propose to use the ARD _Matérn 5/2 kernel_ 
 How it look like (5) # polycop

=> With this covariance function we can obtain **twice_differentiable sample functions**.( result correspond to quasi-newton
Method )

##### Manage the hyperparameters that control the covariance function ( != hyperparameters being optimized ) :

=> They propose to use D + 3 Gaussian process hyperparameters that is more appropriate for the problem.
 How it look like # polycop
 

##### Result :

=> For fully-Bayesian treatment of hyperparameters, they propose **to marginalize over hyperparameters and compute
the integrated acquisition function**
 How it look like (6) # polycop
 
 ###### Advantages :
 
+It takes in consideration the uncertainty for hyperparameters for EI and probability of improvement.
+It's possible to have a Monte Carlo estimate of the integrated EI. [What is Monte Carlo ?](https://www.youtube.com/watch?v=AyBNnkYrSWY)
+It's computationally dominated by the cubic cost of solving an N-dimensional linear system (cubic complexity).



### 3.2 Modeling Costs ( solve LIMITATION 2 )

  
#### Ultimate Target :
=> **find a good setting of our hyperparameters as quickly as possible**.


#### Problem ?
=> Acquisition procedures as EI try to make the best progress possible in the next function evaluation but
don't consider the progress in term of **execution time** which is very important in practical point of view.


#### Solution ?
To improve performance in terms of wallclock time => They propose **optimizing with the EI per second**.


#### Principle ?
Acquire points that are not only likely to be good but are also likely to be evaluated quickly.


#### How ?
=> Use unknown objective function **f(x)** and unknown duration **c(x)** then **model ln c(x) alongside f(x)**
with GP machinery, then by assuming that these function are independant we can **easily compute the predicted
expected inverse duration** then use it to **compute the EI per second**.









