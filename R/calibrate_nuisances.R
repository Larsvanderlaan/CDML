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
calibrate_inverse_weights <- function(A, pi1, pi0, weights) {

  if(missing(weights)){
    weights <- rep(1,length(A))
  }
  # Calibrate pi1 using monotonic XGBoost
  calibrator_pi1 <- isoreg_with_xgboost(pi1, A, weights = weights)
  pi1_star <- calibrator_pi1(pi1)

  # Set minimum truncation level for treated group
  c1 <- min(pi1_star[A == 1])
  pi1_star <- pmax(pi1_star, c1)
  alpha1_star <- 1 / pi1_star

  # Calibrate pi0 using monotonic XGBoost
  calibrator_pi0 <- isoreg_with_xgboost(pi0, 1 - A, weights = weights)
  pi0_star <- calibrator_pi0(pi0)

  # Set minimum truncation level for control group
  c0 <- min(pi0_star[A == 0])
  pi0_star <- pmax(pi0_star, c0)
  alpha0_star <- 1 / pi0_star

  # Return calibrated inverse weights for both groups
  return(list(alpha1_star = alpha1_star, alpha0_star = alpha0_star, pi1_star = pi1_star, pi0_star = pi0_star))
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
calibrate_outcome_regression <- function(Y, mu1, mu0, A, weights) {
  if(missing(weights)){
    weights <- rep(1,length(A))
  }
  # Calibrate mu1 using monotonic XGBoost for treated group
  calibrator_mu1 <- isoreg_with_xgboost(mu1[A==1], Y[A==1], weights = weights[A==1])
  mu1_star <- calibrator_mu1(mu1)

  # Calibrate mu0 using monotonic XGBoost for control group
  calibrator_mu0 <- isoreg_with_xgboost(mu0[A==0], Y[A==0], weights = weights[A==0])
  mu0_star <- calibrator_mu0(mu0)

  # Return calibrated values for both groups
  return(list(mu1_star = mu1_star, mu0_star = mu0_star))
}
