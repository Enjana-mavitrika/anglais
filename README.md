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
## 3 PRACTICAL CONSIDERATIONS FOR BAYESIAN OPTIMIZATION OF HYPERPARAMETERS


  
  
