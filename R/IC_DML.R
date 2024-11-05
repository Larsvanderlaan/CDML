#' Calibrate Inverse Weights
#'
#' Calibrates inverse weights using isotonic regression with XGBoost for two propensity scores.
#'
#' @param A A binary indicator variable.
#' @param pi1 Cross-fitted propensity score estimates for the treatment group (\code{A = 1}).
#' @param pi0 Cross-fitted propensity score estimates for the control group (\code{A = 0}).
#'
#' @return A list containing calibrated inverse weights for each group:
#' \describe{
#'   \item{alpha1_star}{Calibrated inverse weights for \code{A = 1}.}
#'   \item{alpha0_star}{Calibrated inverse weights for \code{A = 0}.}
#' }
#'
#' @details
#' This function applies monotonic XGBoost calibration to the estimated propensity scores
#' for both the treatment and control groups, ensuring isotonic regression calibration.
#'
#' @export
calibrate_inverse_weights <- function(A, pi1, pi0) {

  # Calibrate pi1 using monotonic XGBoost
  calibrator_pi1 <- isoreg_with_xgboost(pi1, A)
  pi1_star <- calibrator_pi1(pi1)

  # Set minimum truncation level for treated group
  c1 <- min(pi1_star[A == 1])
  pi1_star <- pmax(pi1_star, c1)
  alpha1_star <- 1 / pi1_star

  # Calibrate pi0 using monotonic XGBoost
  calibrator_pi0 <- isoreg_with_xgboost(pi0, 1 - A)
  pi0_star <- calibrator_pi0(pi0)

  # Set minimum truncation level for control group
  c0 <- min(pi0_star[A == 0])
  pi0_star <- pmax(pi0_star, c0)
  alpha0_star <- 1 / pi0_star

  # Return calibrated inverse weights for both groups
  return(list(alpha1_star = alpha1_star, alpha0_star = alpha0_star))
}


#' Calibrate Outcome Regression Predictions
#'
#' Calibrates outcome regression predictions using isotonic regression with XGBoost.
#'
#' @param Y Observed outcomes.
#' @param mu1 Cross-fitted predicted outcome for the treated group.
#' @param mu0 Cross-fitted predicted outcome for the control group.
#' @param A A binary indicator variable.
#'
#' @return A list containing calibrated predictions for each group:
#' \describe{
#'   \item{mu1_star}{Calibrated predictions for \code{A = 1}.}
#'   \item{mu0_star}{Calibrated predictions for \code{A = 0}.}
#' }
#'
#' @details
#' This function calibrates outcome regression predictions for both treated and control groups
#' using monotonic XGBoost, which applies isotonic regression constraints.
#'
#' @export
calibrate_outcome_regression <- function(Y, mu1, mu0, A) {
  # Calibrate mu1 using monotonic XGBoost for treated group
  calibrator_mu1 <- isoreg_with_xgboost(mu1[A==1], Y[A==1])
  mu1_star <- calibrator_mu1(mu1)

  # Calibrate mu0 using monotonic XGBoost for control group
  calibrator_mu0 <- isoreg_with_xgboost(mu0[A==0], Y[A==0])
  mu0_star <- calibrator_mu0(mu0)

  # Return calibrated values for both groups
  return(list(mu1_star = mu1_star, mu0_star = mu0_star))
}



#' Compute Calibrated Debiased Machine Learning Estimate (ICDML)
#'
#' Calculates the Calibrated Debiased Machine Learning (C-DML) estimator for
#' causal inference, specifically for estimating linear functionals of the outcome regression.
#'
#' @param A A binary indicator variable (1 for treatment, 0 for control).
#' @param Y Observed outcomes.
#' @param mu1 Cross-fitted outcome regression estimates for the treated group.
#' @param mu0 Cross-fitted outcome regression estimates for the control group.
#' @param pi1 Cross-fitted propensity score estimates for the treatment group (\code{A = 1}).
#' @param pi0 Cross-fitted propensity score estimates for the control group (\code{A = 0}).
#'
#' @return A numeric value representing the estimated causal effect, \eqn{\tau_n}, using the ICDML estimator.
#'
#' @details
#' This function implements the ICDML estimator, a novel calibrated debiased machine learning
#' approach designed for estimating causal effects in a doubly robust manner.
#' Doubly robust consistency and inference are achieved through a combination of cross-fitting,
#' isotonic calibration, and debiased machine learning.
#'
#' The ICDML estimator is part of the C-DML framework, which leverages the link between calibration
#' and doubly robust asymptotic linearity. By integrating outcome regression calibration
#' and inverse propensity weights, ICDML maintains asymptotic linearity when either the
#' outcome regression or the Riesz representer is sufficiently well-estimated, even if the other
#' is inconsistently or slowly estimated. This robustness enables valid inference with flexible
#' machine learning models.
#'
#' For confidence intervals, a bootstrap-assisted approach is recommended to maintain doubly
#' robust inference properties.
#'
#' @seealso \code{\link{calibrate_outcome_regression}}, \code{\link{calibrate_inverse_weights}}
#'
#' @export
estimate_IC_DML <- function(A, Y, mu1, mu0, pi1, pi0) {
  # Calibrate outcome regression predictions
  calibrated_outcome <- calibrate_outcome_regression(Y, mu1, mu0, A)
  mu1_star <- calibrated_outcome$mu1_star
  mu0_star <- calibrated_outcome$mu0_star

  # Calibrate inverse propensity weights
  calibrated_weights <- calibrate_inverse_weights(A, pi1, pi0)

  # Compute ICDML estimate
  mu_star <- ifelse(A == 1, mu1_star, mu0_star)
  alpha_n <- ifelse(A == 1, calibrated_weights$alpha1_star, -calibrated_weights$alpha0_star)
  tau_n <- mean(mu1_star - mu0_star + alpha_n * (Y - mu_star))

  return(tau_n)
}


#' Bootstrap Calibrated Debiased Machine Learning Estimate (ICDML)
#'
#' Computes a bootstrap-assisted version of the ICDML estimator to obtain a confidence interval
#' for the causal effect, \eqn{\tau_n}, leveraging calibrated debiased machine learning.
#'
#' @param A A binary indicator variable (1 for treatment, 0 for control).
#' @param Y Observed outcomes.
#' @param mu1 Predicted outcome for the treated group.
#' @param mu0 Predicted outcome for the control group.
#' @param pi1 Estimated propensity score for the treatment group (\code{A = 1}).
#' @param pi0 Estimated propensity score for the control group (\code{A = 0}).
#' @param nboot Integer. Number of bootstrap samples (default is 1000).
#' @param folds List or \code{NULL}. A list of cross-validation folds, where each fold contains a validation set of indices. If \code{NULL}, a simple random sampling is used for each bootstrap iteration.
#' @param alpha Significance level for the confidence interval (default is 0.05).
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{tau_n}{Estimated causal effect using ICDML.}
#'   \item{CI}{Bootstrap confidence interval for \code{tau_n}.}
#' }
#'
#' @details
#' This function estimates the causal effect using the ICDML estimator and computes a bootstrap confidence
#' interval by resampling within cross-validation folds. If \code{folds} is \code{NULL}, a simple random sample is drawn for each bootstrap iteration. Cross-fitting with bootstrapping allows for doubly robust inference, even when some nuisance function estimates are inconsistent or slow.
#'
#' The confidence interval is constructed by calculating the empirical quantiles of the bootstrap distribution
#' of \code{tau_n}. The interval reflects the uncertainty of the ICDML estimate.
#'
#' @seealso \code{\link{estimate_IC_DML}}, \code{\link{calibrate_outcome_regression}}, \code{\link{calibrate_inverse_weights}}
#'
#' @examples
#' \dontrun{
#' # Example usage
#' set.seed(123)
#' A <- rbinom(100, 1, 0.5)
#' Y <- rnorm(100, mean = A)
#' mu1 <- rep(1, 100)
#' mu0 <- rep(0, 100)
#' pi1 <- rep(0.5, 100)
#' pi0 <- rep(0.5, 100)
#' result <- estimate_IC_DML_bootstrap(A, Y, mu1, mu0, pi1, pi0, nboot = 500, alpha = 0.05)
#' print(result$tau_n)
#' print(result$CI)
#' }
#'
#' @export
estimate_IC_DML_bootstrap <- function(A, Y, mu1, mu0, pi1, pi0, nboot = 1000, folds = NULL, alpha = 0.05) {
  # Load data
  data <- data.table::data.table(A, Y, mu1, mu0, pi1, pi0)

  # Calculate the ICDML point estimate
  tau_n <- estimate_IC_DML(A, Y, mu1, mu0, pi1, pi0)

  # Perform bootstrap resampling
  bootstrap_estimates <- sapply(1:nboot, function(iter) {
    tryCatch({
      # Determine bootstrap indices
      if (is.null(folds)) {
        # Random sample when folds are not provided
        bootstrap_indices <- sample(seq_len(nrow(data)), nrow(data), replace = TRUE)
      } else {
        # Sample within each fold if folds are provided
        bootstrap_indices <- unlist(lapply(folds, function(fold) {
          sample(fold, length(fold), replace = TRUE)
        }))
      }

      # Subset data based on bootstrap indices
      data_boot <- data[bootstrap_indices,]
      tau_boot <- estimate_IC_DML(data_boot$A, data_boot$Y,
                                  mu1 = data_boot$mu1, mu0 = data_boot$mu0,
                                  pi1 = data_boot$pi1, pi0 = data_boot$pi0)
      return(tau_boot)
    }, error = function(e) NA) # Return NA if an error occurs
  })

  # Remove NA values from bootstrap estimates
  bootstrap_estimates <- na.omit(bootstrap_estimates)

  # Calculate the confidence interval
  CI <- quantile(bootstrap_estimates, probs = c(alpha / 2, 1 - alpha / 2), na.rm = TRUE)

  return(list(tau_n = tau_n, CI = CI))
}
