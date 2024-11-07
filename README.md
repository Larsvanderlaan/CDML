# CDML Package

## Overview

The CDML (Calibrated Debiased Machine Learning) package implements tools for automatic doubly robust inference for linear functionals in causal inference problems. Many causal estimands of interest, such as average causal effects of static, dynamic, and stochastic interventions, can be expressed as linear functionals of the outcome regression function. CDML leverages modern machine learning to provide debiased estimators that are doubly robust and asymptotically linear. This enables consistent estimation as well as valid statistical inference, including the construction of confidence intervals and hypothesis testing.

The package is inspired by and implements methodologies from our paper on [Automatic Doubly Robust Inference for Linear Functionals via Calibrated Debiased Machine Learning](https://arxiv.org/pdf/2411.02771v1).

## Abstract

In causal inference, many estimands of interest can be expressed as a linear functional of the outcome regression function; this includes, for example, average causal effects of static, dynamic and stochastic interventions. For learning such estimands, in this work, we propose novel debi- ased machine learning estimators that are doubly robust asymptotically linear, thus providing not only doubly robust consistency but also facilitating doubly robust inference (e.g., confidence intervals and hypothesis tests). To do so, we first establish a key link between calibration, a ma- chine learning technique typically used in prediction and classification tasks, and the conditions needed to achieve doubly robust asymptotic linearity. We then introduce calibrated debiased machine learning (C-DML), a unified framework for doubly robust inference, and propose a specific C-DML estimator that integrates cross-fitting, isotonic calibration, and debiased ma- chine learning estimation. A C-DML estimator maintains asymptotic linearity when either the outcome regression or the Riesz representer of the linear functional is estimated sufficiently well, allowing the other to be estimated at arbitrarily slow rates or even inconsistently. We propose a simple bootstrap-assisted approach for constructing doubly robust confidence intervals. Our theoretical and empirical results support the use of C-DML to mitigate bias arising from the inconsistent or slow estimation of nuisance functions.

## Installation

To install the CDML package, you can use pip:

```R
devtools::install_github("Larsvanderlaan/cdml")"
