#' Default Learners List
#'
#' This list defines a set of default learners for use in machine learning tasks.
#' The learners include generalized linear models, multivariate adaptive regression splines,
#' generalized additive models, random forests, and boosted trees with early stopping.
#'
#' @import sl3
#' @importFrom glmnet glmnet
#' @importFrom earth earth
#' @importFrom mgcv gam
#' @importFrom ranger ranger
#' @importFrom xgboost xgboost
#' @return A list of instantiated learners from various algorithms.
default_learners <- function(){ list(

  # Lrnr_glmnet: Generalized Linear Model (GLM) with elastic net regularization.
  # Suitable for linear and logistic regression with built-in regularization.
  Lrnr_glmnet$new(),

  # Lrnr_earth: Multivariate Adaptive Regression Splines (MARS) learner.
  # Degree 2 allows for modeling interactions up to second order.
  Lrnr_earth$new(degree = 2),

  # Lrnr_gam: Generalized Additive Model (GAM) learner for flexible nonlinear modeling.
  Lrnr_gam$new(),

  # Lrnr_ranger: Random forest learner with a specified maximum tree depth.
  # Here, max.depth = 8 limits the depth of trees to control complexity.
  Lrnr_ranger$new(max.depth = 8),

  # Lrnr_xgboost_early_stopping: XGBoost learner with early stopping for boosting.
  # Configured with different maximum depths to control tree complexity.
  Lrnr_xgboost_early_stopping$new(max_depth = 3),
  Lrnr_xgboost_early_stopping$new(max_depth = 4),
  Lrnr_xgboost_early_stopping$new(max_depth = 5)
)
}
