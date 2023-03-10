---
title: 4 Hyperparameters Optimization Frameworks for Machine Learning
author: Thomas Shaw
date: '2023-03-09'
slug: []
categories:
  - machine learning
  - python
tags: []
subtitle: ''
excerpt: Choosing the right hyperparameters can make or break a model’s performance.
draft: no
series: ~
layout: single
---

Machine learning models rely heavily on hyperparameters, which are essentially the configuration settings that control the performance and the behavior of the model. Properly tuning these hyperparameters can lead to significant improvements in the model's accuracy and overall performance. However, manual tuning of hyperparameters is hard can be a time-consuming task, often requiring many iterations and experimentation. That's where hyperparameter optimization frameworks come into play, which help automate the process of finding the optimal set of hyperparameters for a machine learning model. In this blog post, we will explore some of the most popular hyperparameter optimization frameworks available, including Scikit-learn, Optuna, Hyperopt, and Ray Tune. This post will provide an overview of each framework and demonstrate how they can be used to optimize the hyperparameters for machine learning models.

### 1. Scikit-learn

Scikit-learn is a popular machine learning library in Python that includes a `GridSearchCV` class for grid search and `RandomizedSearchCV` class for random search. These classes can be used to perform hyperparameter tuning for a wide range of machine learning algorithms, including regression, classification, and clustering. 

One of the key features of Scikit-learn is its ease of use. It provides a simple and consistent interface for performing various machine learning tasks. Scikit-learn is also well-documented and has a large user community, which makes it easy to find help and examples when needed.

### 2. Optuna

Optuna is a powerful hyperparameter optimization library that uses a Bayesian approach to model the objective function. It supports various search algorithms, including grid search, random search, TPE, and CMA-ES and can be used for any machine learning framework, including TensorFlow, PyTorch, and scikit-learn. Optuna is designed to be easy to use and provides a convenient interface for defining search spaces and objective functions.

One of the key features of Optuna is its flexibility. It provides a wide range of search algorithms and can optimize any user-defined objective function. Optuna also includes utilities for parallelizing the optimization process, which can speed up the search for the best set of hyperparameters.

### 3. Hyperopt

Hyperopt is a Python library for hyperparameter optimization that uses the Tree-structured Parzen Estimator (TPE) algorithm. It can be used with any machine learning library and supports both grid and random search. 

One of the key features of Hyperopt is its support for Bayesian optimization. The TPE algorithm used by Hyperopt is a Bayesian optimization algorithm that can efficiently search the hyperparameter space. Hyperopt also provides a simple and consistent interface for defining search spaces and objective functions.

### 4. Ray Tune

Ray Tune is a distributed hyperparameter tuning library that can be used with various machine learning libraries, including TensorFlow, PyTorch, and scikit-learn. It supports various search algorithms, including grid search, random search, and BOHB. Ray Tune is designed to be scalable and can run on large clusters or cloud environments. It also provides a convenient interface for defining search spaces and objective functions.

One of the key features of Ray Tune is its scalability. It can run on large clusters or cloud environments and can efficiently parallelize the optimization process. Ray Tune also supports the BOHB algorithm, which is a state-of-the-art Bayesian optimization algorithm that can efficiently search the hyperparameter space.

## Example

### The Dataset

We'll use the well-known iris dataset, which is included in scikit-learn. This dataset consists of 150 samples with four features: sepal length, sepal width, petal length, and petal width. The goal is to predict the species of each sample, which can be one of three possible values: setosa, versicolor, or virginica.

Let's load the dataset and split it into training and test sets:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)
```

### The Model

We'll use a support vector machine (SVM) classifier as our model. Specifically, we'll use scikit-learn's `SVC`class with a radial basis function (RBF) kernel. 

```python
from sklearn.svm import SVC

def train_svm(C, gamma):
    clf = SVC(C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    return clf
```

This function takes two hyperparameters, `C`and `gamma`, as arguments and trains an SVM classifier on the training data.

### Scikit-learn

Scikit-learn provides several ways to perform hyperparameter optimization. Here, we'll use `GridSearchCV`, which performs an exhaustive search over a specified parameter grid.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 10]}

svm = SVC(kernel='rbf')
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best hyperparameters: {grid_search.best_params_}")
```

We define a `param_grid` dictionary with the values we want to test for `C` and `gamma`. Then, we create an SVM classifier with an RBF kernel and pass it, along with the `param_grid` and the number of cross-validation folds (`cv=5`), to `GridSearchCV`.

### Optuna

```python
import optuna

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e4)

    clf = SVC(C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best hyperparameters: {study.best_params}")
```

Inside the `objective` function, we use the `suggest_loguniform`method of the `trial`object to define the search space for `C`and `gamma`. This method suggests values for the hyperparameters using a log-uniform distribution, which is often a good choice for hyperparameters that can span several orders of magnitude. We then create an Optuna `study`object with the `maximize`direction, which indicates that we want to maximize the objective function (i.e., maximize the test set accuracy).

### Hyperopt

```python
from hyperopt import fmin, tpe, hp

space = {
    'C': hp.loguniform('C', -4, 4),
    'gamma': hp.loguniform('gamma', -4, 4)
}

def objective(params):
    clf = SVC(C=params['C'], gamma=params['gamma'])
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return {'loss': -score, 'status': 'ok'}

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

print(f"Best hyperparameters: {best}")
```

We define a search space dictionary `space` with the `hp.loguniform` method of Hyperopt. This method is similar to Optuna's `suggest_loguniform` method and defines a log-uniform distribution for the hyperparameters. We also define an `objective` function that takes a dictionary of hyperparameters as input and returns a dictionary with a `loss` key (set to the negative accuracy score) and a `status` key set to `'ok'`.

We call the `fmin` function of Hyperopt with the `objective` function, the search space, the `'tpe.suggest'` algorithm (which stands for Tree-structured Parzen Estimator), and the maximum number of evaluations to run (`max_evals=100`). This function performs the hyperparameter search and returns the best hyperparameters.

### Ray Tune

```python
import ray
from ray import tune
from ray.tune import grid_search

def train_svm_tune(config):
    clf = SVC(C=config['C'], gamma=config['gamma'])
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

config = {
    'C': grid_search([0.1, 1, 10, 100]),
    'gamma': grid_search([0.01, 0.1, 1, 10])
}

analysis = tune.run(
    train_svm_tune,
    config=config,
    metric='mean_accuracy',
    mode='max',
    num_samples=100,
    resources_per_trial={'cpu': 1},
    local_dir='./tune_results'
)

print(f"Best hyperparameters: {analysis.best_config}")
```

The `train_svm_tune` function performs the hyperparameter search using Ray Tune and returns an `Analysis` object that contains the results of the search.

We define a configuration dictionary `config` with two hyperparameters, `C` and `gamma`, and use the `grid_search` function of Ray Tune to define a discrete search space for each hyperparameter.

We then call the `tune.run` function of Ray Tune with the `train_svm_tune` function, the configuration dictionary, the `metric` parameter set to `'mean_accuracy'` (which tells Ray Tune to use the mean of the accuracy scores as the optimization metric), the `mode` parameter set to `'max'` (which indicates that we want to maximize the optimization metric), the `num_samples` parameter set to `100` (which sets the number of trials to run), the `resources_per_trial` parameter set to `{'cpu': 1}` (which specifies the amount of CPU resources to allocate to each trial), and the `local_dir` parameter set to `'./tune_results'` (which specifies the directory to store the results).