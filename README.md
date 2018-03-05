# anglais


## (suite)<br/>
Begstras'team worked on different strategises for optimizing the hyperparameter of machine learning algorithms.
They conclude that grid search strategies are inferior to random search. So they suggested to use Gaussian
process Bayesian optimization, and proposed the **Tree Parzen Algorithm**.
(Récemment, Bergstra a exploré différentes stratégies pour optimiser les hyperparamètres des algorithmes d'apprentissage automatique. 
Ils ont démontré que les stratégies de recherche de grille sont inférieures à la recherche aléatoire, et ont suggéré l'utilisation de l'optimisation bayésienne de processus gaussien, 
en optimisant les hyperparamètres d'une covariance carré-exponentielle, et ont proposé l'algorithme de Tree Parzen.)

## 2) BAYESIAN OPTIMIZATION WITH GAUSSIAN PROCESS PRIORS

In Bayesian optimization as in other type of optimization, the target is **finding the minimum of a function
f(x)** amoung some boundset X, and X is a subset of R^D.
(Comme dans les autres types d'optimisation, dans l'optimisation bayésienne, nous cherchons à trouver le minimum d'une fonction f (x) sur un ensemble borné
X, que nous prendrons pour être un sous-ensemble de R (D))

The difference between Bayesian optimization and other strategies, is that it generate a probalistic
model for f(x) and use it to decide amoung X, where to make the next evaluation while taking in consideration
uncertainty(Probabilistic).
(Ce qui différencie l'optimisation bayésienne des autres procédures, c'est qu'elle construit un modèle probabiliste 
pour f (x), puis exploite ce modèle pour prendre des décisions sur l'endroit où
X pour évaluer ensuite la fonction, tout en intégrant l'incertitude.)

The essential principle is to use all the information given by the previous evaluation of f(x) and not 
consider only __local gradient__ and __Hessian approximations__ .
(la philosophie essentielle est d'utiliser toutes les informations disponibles à partir des évaluations 
précédentes de f (x) et de ne pas se fier uniquement au gradient local et aux approximations de Hess.)

As a result of this strategy, it's possible to find the minimum of a complex non-convex functions with
only few evaluations. But it cost more computation to determine the next point to try.
(Il en résulte une procédure qui peut trouver le minimum de fonctions non-convexes difficiles avec 
relativement peu d'évaluations, au prix d'effectuer plus de calculs pour déterminer le prochain point à essayer.)

But evaluations of f(x) are often expensive when it requires training a machine learning algorithm
some extra computation is not a big deal if it help to make better decisions.
(Lorsque les évaluations de f (x) sont coûteuses à réaliser - comme c'est le cas quand il faut former un algorithme 
d'apprentissage automatique - alors il est facile de justifier un calcul supplémentaire pour prendre de meilleures décisions)

When performing Bayesian optimization there are **2 major choices to make**:
  - 1. select amoung previous functions, functions that will give assumptions(more informations) about the function being
  optimized. And for this, they choose **Gaussian Process Priors** that is more efficient performing such task.
  
  - 2. choose an acquisition function to generate a utility function from previous model, in order to find
  the next point to evaluate.
  (Deuxièmement, nous devons choisir une fonction d'acquisition, qui est utilisée pour construire une fonction 
  d'utilité à partir du modèle postérieur, ce qui nous permet de déterminer le point suivant à évaluer.)
