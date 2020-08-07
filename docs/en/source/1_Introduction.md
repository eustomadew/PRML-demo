# 1 Introduction

## 1.1 Example: Polynomial Curve Fitting

**A training set** comprising $N$ observations:
- data $x$, i.e., $\mathbf{x}= (x_1, ..., x_N)^\mathsf{T}$
- target value $t$

Fit the data using a polynomial function of the form
$$y(x,\mathbf{w}) = w_0 + w_1x + w_2x^2 + ... + w_Mx^M = \sum_{j=0}^M w_jx^j$$
where
- $M$ is the *order* of the polynomial
- $x_j$ denotes $x$ raised to the power of $j$
- $w_j$ is polynomial coefficients

*Note that*, although the polynomial function $y(x,\mathbf{w})$ is a nonlinear function of $x$, it is a linear function of the coefficients $\mathbf{w}$ (vector).  
Functions, such as the polynomial, which are linear in the unknown parameters have important properties and are called *linear models*.

**Coefficients**: determined by fitting the polynomial to the training data, e.g., by minimizing an *error function* that measures the misfit between the function $y(x,\mathbf{w})$, for any given value of $\mathbf{w}$ and the training set data points.

$$E(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N \{y(x_n,\mathbf{w}) - t_n\}^2$$

## 1.2 Probability Theory

### 1.2.1 Probability densities

### 1.2.2 Expectations and covariances

## 1.3 Model Selection

## 1.4 The Curse of Dimensionality

## 1.5 Decision Theory

## 1.6 Information Theory

## Exercises