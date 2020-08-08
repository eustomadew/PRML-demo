==============
Introduction
==============


Example: Polynomial Curve Fitting
====================================

**A training set** comprising `N` observations:

- data :math:`x`, i.e., :math:`\mathbf{x}= (x_1, ..., x_N)^\mathsf{T}`
- target value :math:`t`

**A polynomial function**, to fit the data where: 
:math:`\displaystyle y(x,\mathbf{w}) = w_0 + w_1x + w_2x^2 + ... + w_Mx^M = \sum_{j=0}^M w_jx^j` 

- :math:`M` is the *order* of the polynomial
- :math:`x_j` denotes :math:`x` raised to the power of :math:`j`
- :math:`w_j` is polynomial coefficients

*Note that*, although the polynomial function :math:`y(x,\mathbf{w})` is a nonlinear function of :math:`x`, it is a linear function of the coefficients :math:`\mathbf{w}` (vector).  
Functions, such as the polynomial, which are linear in the unknown parameters have important properties and are called *linear models*.

**Coefficients**: determined by fitting the polynomial to the training data, e.g., by minimizing an *error function* that measures the misfit between the function :math:`y(x,\mathbf{w})`, for any given value of :math:`\mathbf{w}` and the training set data points.

Minimizing an **error function**: 
:math:`\displaystyle E(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N \{y(x_n,\mathbf{w}) - t_n\}^2`


Probability Theory
=====================

-------------------------
Probability densities
-------------------------

--------------------------------
Expectations and covariances
--------------------------------


Model Selection
==================


The Curse of Dimensionality
==============================

Decision Theory
==================

## Information Theory

## Exercises
