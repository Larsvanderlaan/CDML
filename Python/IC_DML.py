import xgboost as xgb
import numpy as np
from isocal import  *


def calibrate_inverse_weights(A, pi1, pi0):
    """
    Calibrates inverse weights using isotonic regression with XGBoost for two propensity scores.

    Args:
        A (np.array): Binary indicator variable.
        pi1 (np.array): Propensity score for treatment group (A = 1).
        pi0 (np.array): Propensity score for control group (A = 0).

    Returns:
        dict: Contains calibrated inverse weights:
              - alpha1_star: Inverse weights for A = 1.
              - alpha0_star: Inverse weights for A = 0.
    """
    calibrator_pi1 = isoreg_with_xgboost(pi1, A)
    pi1_star = calibrator_pi1(pi1)
    c1 = np.min(pi1_star[A == 1])
    pi1_star = np.maximum(pi1_star, c1)
    alpha1_star = 1 / pi1_star

    calibrator_pi0 = isoreg_with_xgboost(pi0, 1 - A)
    pi0_star = calibrator_pi0(pi0)
    c0 = np.min(pi0_star[A == 0])
    pi0_star = np.maximum(pi0_star, c0)
    alpha0_star = 1 / pi0_star

    return {'alpha1_star': alpha1_star, 'alpha0_star': alpha0_star}

def calibrate_outcome_regression(Y, mu1, mu0, A):
    """
    Calibrates outcome regression predictions using isotonic regression with XGBoost.

    Args:
        Y (np.array): Observed outcomes.
        mu1 (np.array): Predicted outcome for treated group.
        mu0 (np.array): Predicted outcome for control group.
        A (np.array): Binary indicator variable.
    
    Returns:
        dict: Contains calibrated predictions:
              - mu1_star: Calibrated predictions for A = 1.
              - mu0_star: Calibrated predictions for A = 0.
    """
    calibrator_mu1 = isoreg_with_xgboost(mu1, A)
    mu1_star = calibrator_mu1(mu1)

    calibrator_mu0 = isoreg_with_xgboost(mu0, 1 - A)
    mu0_star = calibrator_mu0(mu0)

    return {'mu1_star': mu1_star, 'mu0_star': mu0_star}

def estimate_IC_DML(A, Y, mu1, mu0, pi1, pi0):
    """
    Computes the ICDML estimator for causal inference, estimating linear functionals
    of the outcome regression.

    Args:
        A (np.array): Binary indicator variable (1 for treatment, 0 for control).
        Y (np.array): Observed outcomes.
        mu1 (np.array): Predicted outcome for the treated group.
        mu0 (np.array): Predicted outcome for the control group.
        pi1 (np.array): Propensity score for the treatment group (A = 1).
        pi0 (np.array): Propensity score for the control group (A = 0).

    Returns:
        float: Estimated causal effect, tau_n, using the ICDML estimator.
    """
    calibrated_outcome = calibrate_outcome_regression(Y, mu1, mu0, A)
    mu1_star = calibrated_outcome['mu1_star']
    mu0_star = calibrated_outcome['mu0_star']

    calibrated_weights = calibrate_inverse_weights(A, pi1, pi0)
    mu_star = np.where(A == 1, mu1_star, mu0_star)
    alpha_n = np.where(A == 1, calibrated_weights['alpha1_star'], -calibrated_weights['alpha0_star'])
    tau_n = np.mean(mu1_star - mu0_star + alpha_n * (Y - mu_star))
    
    return tau_n

def estimate_IC_DML_bootstrap(A, Y, mu1, mu0, pi1, pi0, nboot=1000, folds=None, alpha=0.05):
    """
    Computes a bootstrap-assisted version of the ICDML estimator to obtain a confidence interval
    for the causal effect, leveraging calibrated debiased machine learning.
    
    Args:
        A (np.array): Binary indicator variable (1 for treatment, 0 for control).
        Y (np.array): Observed outcomes.
        mu1 (np.array): Predicted outcome for the treated group.
        mu0 (np.array): Predicted outcome for the control group.
        pi1 (np.array): Propensity score for the treatment group (A = 1).
        pi0 (np.array): Propensity score for the control group (A = 0).
        nboot (int): Number of bootstrap samples (default is 1000).
        folds (list or None): List of cross-validation folds, where each fold contains a validation set 
                              of indices. If None, a simple random sampling is performed.
        alpha (float): Significance level for the confidence interval (default is 0.05).
    
    Returns:
        dict: Contains the estimated causal effect and confidence interval:
              - tau_n: Estimated causal effect using ICDML.
              - CI: Bootstrap confidence interval for tau_n.
    """
    # Calculate the ICDML point estimate
    tau_n = estimate_IC_DML(A, Y, mu1, mu0, pi1, pi0)
    
    # Perform bootstrap resampling
    bootstrap_estimates = []
    for _ in range(nboot):
        try:
            if folds is None:
                # Random sampling when folds are not provided
                bootstrap_indices = np.random.choice(len(A), size=len(A), replace=True)
            else:
                # Sample within each fold if folds are provided
                bootstrap_indices = np.concatenate([
                    np.random.choice(fold, size=len(fold), replace=True) for fold in folds
                ])
            
            # Subset data based on bootstrap indices
            data_boot = (A[bootstrap_indices], Y[bootstrap_indices], 
                         mu1[bootstrap_indices], mu0[bootstrap_indices], 
                         pi1[bootstrap_indices], pi0[bootstrap_indices])
            
            # Compute ICDML for bootstrap sample
            tau_boot = estimate_IC_DML(*data_boot)
            bootstrap_estimates.append(tau_boot)
        except Exception:
            continue
    
    # Convert bootstrap estimates to array and compute confidence interval
    bootstrap_estimates = np.array(bootstrap_estimates)
    CI = np.quantile(bootstrap_estimates, [alpha / 2, 1 - alpha / 2])
    
    return {'tau_n': tau_n, 'CI': CI}
