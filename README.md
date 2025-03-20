# Deep Learning (CW1)

This repository contains two primary tasks for Deep Learning

- **Task 1:** Optimising Logistic Binary Regression Models  
- **Task 2:** Regularising Extreme Learning Machines (ELMs)

Each task has **two** scripts:
- **Task 1**: `task1/task.py` and `task1/task1a.py`
- **Task 2**: `task2/task.py` and `task2/task2a.py`

Below is an overview of these scripts, followed by instructions on running them and details on the environment.

---

## Table of Contents

1. [Task 1 Overview](#task-1-overview)
   - [task.py](#taskpy-1)
   - [task1a.py](#task1apy)
2. [Task 2 Overview](#task-2-overview)
   - [task.py](#taskpy-2)
   - [task2a.py](#task2apy)

---

## Task 1 Overview

### task.py
- **Description**:  
  Implements logistic regression with polynomial feature expansion for binary classification.  
  - Generates a synthetic dataset of 200 training points and 100 test points with random noise.  
  - Trains the model via stochastic gradient descent (SGD) using either:
    - **Cross-Entropy** loss, or
    - **Root-Mean-Square** (RMSE) loss.
  - Compares different polynomial orders (\(M = 1,2,3\)) and prints metrics (like accuracy) on both training and test sets.
  - Includes a short commentary on classification metrics and differences in train/test performance.

### task1a.py
- **Description**:  
  Extends the logistic regression approach by treating the polynomial order \(M\) as a **learnable** parameter.  
  - Introduces a custom gating mechanism that “activates” or “deactivates” polynomial terms based on a trainable scalar.  
  - Trains this variant via SGD, printing the final learned \(M\) and comparing accuracy to fixed-order models.

---

## Task 2 Overview

### task.py
- **Description**:  
  Implements **Extreme Learning Machines (ELMs)** for multiclass image classification (using the CIFAR-10 dataset).  
  - **Fixed Convolutional Layer**: Weights are randomly initialized and never updated.  
  - **Trainable Fully-Connected Layer**: Projects convolutional features to class logits.
  - Explores **regularization techniques**:
    1. **MyMixUp**: Data augmentation mixing pairs of images/labels.  
    2. **MyEnsembleELM**: Trains multiple ELMs independently and averages their predictions.
  - Trains four variants:
    1. Baseline (no regularization)  
    2. MixUp only  
    3. Ensemble only  
    4. Ensemble + MixUp
  - Reports accuracy and macro F1 over test data, and saves a results montage (`result.png`).

### task2a.py
- **Description**:  
  Replaces the **SGD** training of the ELM with a **direct least-squares** solver (`fit_elm_ls`).  
  - Compares speed and performance (accuracy, etc.) between **SGD** and **LS** on the same ELM architecture.  
  - Implements a **random hyperparameter search** to find better ELM settings (convolutional layer size, kernel size, initialization standard deviation).  
  - Saves a montage of predictions from the best LS-based model as `new_result.png`.

