#' Isotonic Regression with XGBoost
#'
#' Fits an isotonic regression model using XGBoost with monotonic constraints.
#'
#' @param x A vector or matrix of predictor variables.
#' @param y A vector of response variables.
#' @param max_depth Integer. Maximum depth of the trees in XGBoost (default is 15).
#' @param min_child_weight Numeric. Minimum sum of instance weights (Hessian) needed in a child node (default is 20).
#'
#' @return A function that takes a new predictor variable \code{x} and returns the model's predicted values.
#'
#' @details
#' This function uses XGBoost to fit a monotonic increasing model to the data, enforcing isotonic regression
#' through the use of monotonic constraints. The model is trained with one boosting round to achieve a fit
#' that is interpretable as an isotonic regression.
#'
#' @examples
#' \dontrun{
#' # Example data
#' x <- matrix(rnorm(100), ncol = 1)
#' y <- sort(rnorm(100))
#'
#' # Fit the model
#' iso_model <- isoreg_with_xgboost(x, y)
#'
#' # Predict on new data
#' x_new <- matrix(rnorm(10), ncol = 1)
#' predictions <- iso_model(x_new)
#' }
#'
#' @export
isoreg_with_xgboost <- function(x, y, max_depth = 15, min_child_weight = 20) {
  # Create an XGBoost DMatrix object from the data
  data <- xgboost::xgb.DMatrix(data = as.matrix(x), label = as.vector(y))

  # Set parameters for the monotonic XGBoost model
  params <- list(
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    monotone_constraints = 1,  # Enforce monotonic increase
    eta = 1,
    gamma = 0,
    lambda = 0
  )

  # Train the model with one boosting round
  iso_fit <- xgboost::xgb.train(params = params, data = data, nrounds = 1)

  # Prediction function for new data
  fun <- function(x) {
    data_pred <- xgboost::xgb.DMatrix(data = as.matrix(x))
    pred <- predict(iso_fit, data_pred)
    return(pred)
  }

  return(fun)
}
