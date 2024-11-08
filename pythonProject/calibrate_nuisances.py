import xgboost as xgb
import numpy as np
from isotonic_regression import  *


def calibrate_inverse_probability_weights(A, pi_mat, weights = None, treatment_levels=None):
    """
    Calibrates inverse probability weights using isotonic regression with XGBoost
    for multiple propensity scores.

    Args:
        A (np.array): Treatment indicator variable (taking integer values in 0, 1, 2, ...).
        pi_mat (np.array): Matrix of propensity scores, each column corresponding
                           to a different treatment level.
        treatment_levels (list, optional): List of treatment levels to consider.
                                           If None, defaults to all columns in pi_mat.

    Returns:
        dict: Contains calibrated inverse weights and calibrated propensity scores:
              - 'alpha_star_mat': Matrix of inverse weights for each treatment level.
              - 'pi_star_mat': Matrix of calibrated propensity scores for each treatment level.
    """
    if treatment_levels is None:
        treatment_levels = range(pi_mat.shape[1])

    if weights is None:
        weights = np.ones_like(A)

    # Initialize an array to store the calibrated propensity scores
    pi_star_mat = np.zeros_like(pi_mat)

    for idx, a in enumerate(treatment_levels):
        # Create indicator for current treatment level
        A_ind = (A == a).astype(int)

        # Select the corresponding propensity score column
        pi = pi_mat[:, idx]

        # Calibrate propensity scores using isotonic regression with XGBoost
        calibrator_pi = isoreg_with_xgboost(pi, A_ind, weights = weights)
        pi_star = calibrator_pi(pi)

        # Ensure pi_star values are bounded below to prevent division issues
        c1 = np.min(pi_star[A_ind == 1])
        pi_star = np.maximum(pi_star, c1)

        # Store the calibrated propensity scores in the matrix
        pi_star_mat[:, idx] = pi_star

    # Calculate inverse weights as 1 / pi_star
    alpha_star_mat = 1 / pi_star_mat

    return {'alpha_star': alpha_star_mat, 'pi_star': pi_star_mat}


def calibrate_outcome_regression(Y, mu_mat, A, weights = None, treatment_levels=None):
    """
    Calibrates outcome regression predictions using isotonic regression with XGBoost.

    Args:
        Y (np.array): Observed outcomes.
        mu_mat (np.array): Matrix of predicted outcomes, each column corresponding
                           to a different treatment level.
        A (np.array): Treatment indicator variable (taking integer values in 0, 1, 2, ...).
        treatment_levels (list, optional): List of treatment levels to consider.
                                           If None, defaults to all columns in mu_mat.

    Returns:
        dict: Calibrated predictions for each treatment level:
              - 'mu_star_mat': Matrix of calibrated predictions for each treatment level.
    """
    if treatment_levels is None:
        treatment_levels = range(mu_mat.shape[1])

    if weights is None:
        weights = np.ones_like(Y)

    # Initialize an array to store the calibrated predictions
    mu_star_mat = np.zeros_like(mu_mat)

    for idx, a in enumerate(treatment_levels):
        # Create indicator for current treatment level
        A_ind = (A == a)

        # Select the corresponding predicted outcome column
        mu = mu_mat[:, idx]

        # Calibrate predicted outcomes using isotonic regression with XGBoost
        calibrator_mu = isoreg_with_xgboost(mu[A_ind], Y[A_ind], weights = weights[A_ind])
        mu_star = calibrator_mu(mu)

        # Store the calibrated predictions in the matrix
        mu_star_mat[:, idx] = mu_star

    return {'mu_star': mu_star_mat}


