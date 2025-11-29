# RUM-NN: Neural Estimation Framework for Discrete Choice Models

## Overview
RUM-NN is a Python library designed to model discrete choice behavior using a neural network framework that is fully consistent with Random Utility Maximization (RUM) theory. Unlike traditional models such as Multinomial Logit (MNL) or Multinomial Probit (MNP), RUM-NN allows for **arbitrary error distributions** and can capture **correlations among alternatives**, offering flexibility and robustness for both synthetic and real-world datasets.

This library supports Linear structure for interpretability.

Run Test.py to showcase the RUM-NN model on a synthetic dataset.

The methodology and experiments are detailed in the paper:  
*A Neural Estimation Framework for Discrete Choice Models with Arbitrary Error Distributions*.

---

## Key Features
- Compatible with **any parametric error distribution** (e.g., Gumbel, Normal, Exponential, Pareto).
- Supports **correlated error terms** using Cholesky decomposition.
- Provides **full behavioural interpretability** in linear mode.
- Extensible to **non-linear utility functions** for complex datasets.

---
