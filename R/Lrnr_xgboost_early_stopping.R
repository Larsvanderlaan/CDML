Lrnr_xgboost_early_stopping <- R6Class(
  classname = "Lrnr_xgboost_early_stopping", inherit = Lrnr_base,
  portable = TRUE, class = TRUE,
  public = list(
    initialize = function(nrounds = 1000, max_depth = 5, learning_rate = 0.1, min_child_weight = 10, early_stopping_rounds = 10, p_cv = 0.2, nthread = 1, ...) {
      params <- sl3:::args_to_list()
      super$initialize(params = params, ...)
    },
    importance = function(...) {
      self$assert_trained()

      args <- list(...)
      args$model <- self$fit_object

      importance_result <- sl3:::call_with_args(xgboost::xgb.importance, args)
      rownames(importance_result) <- importance_result[["Feature"]]
      return(importance_result)
    }
  ),
  private = list(
    .properties = c(
      "continuous", "binomial", "categorical", "weights",
      "offset", "importance"
    ),
    .train = function(task) {
      args <- self$params
      p_cv <- args$p_cv # Extract cross-validation proportion

      verbose <- args$verbose
      if (is.null(verbose)) {
        verbose <- getOption("sl3.verbose")
      }
      args$verbose <- as.integer(verbose)

      outcome_type <- self$get_outcome_type(task)
      Y <- outcome_type$format(task$Y)
      if (outcome_type$type == "categorical") {
        Y <- as.numeric(Y) - 1
      }

      Xmat <- as.matrix(task$X)
      if (is.integer(Xmat)) {
        Xmat[, 1] <- as.numeric(Xmat[, 1])
      }
      if (nrow(Xmat) != nrow(task$X) & ncol(Xmat) == nrow(task$X)) {
        Xmat <- t(Xmat)
      }

      # Splitting data into training and validation sets
      set.seed(123) # for reproducibility
      n <- nrow(Xmat)
      train_idx <- sample(seq_len(n), size = round(n * (1 - p_cv)))
      val_idx <- setdiff(seq_len(n), train_idx)

      X_train <- Xmat[train_idx, , drop = FALSE]
      Y_train <- Y[train_idx]
      X_val <- Xmat[val_idx, , drop = FALSE]
      Y_val <- Y[val_idx]

      dtrain <- xgboost::xgb.DMatrix(X_train, label = Y_train)
      dval <- xgboost::xgb.DMatrix(X_val, label = Y_val)

      args$data <- dtrain
      args$watchlist <- list(train = dtrain, eval = dval)

      fit_object <- sl3:::call_with_args(xgboost::xgb.train, args, keep_all = TRUE, ignore = c("formula", "p_cv"))
      best_nrounds <- fit_object$best_iteration

      # Retrain using the full dataset with the selected number of rounds
      full_data <- xgboost::xgb.DMatrix(Xmat, label = Y)
      args$data <- full_data
      args$nrounds <- best_nrounds
      args$watchlist <- NULL # No validation set for final training
      args$early_stopping_rounds <- NULL

      fit_object <- sl3:::call_with_args(xgboost::xgb.train, args, keep_all = TRUE, ignore = "formula", "p_cv")
      fit_object$training_offset <- task$has_node("offset")
      fit_object$link_fun <- if (task$has_node("offset")) args$family$linkfun else NULL

      return(fit_object)
    },
    .predict = function(task = NULL) {
      fit_object <- private$.fit_object

      Xmat <- as.matrix(task$X)
      if (is.integer(Xmat)) {
        Xmat[, 1] <- as.numeric(Xmat[, 1])
      }
      Xmat_ord <- as.matrix(Xmat[, match(fit_object$feature_names, colnames(Xmat))])
      if ((nrow(Xmat_ord) != nrow(Xmat)) & (ncol(Xmat_ord) == nrow(Xmat))) {
        Xmat_ord <- t(Xmat_ord)
      }
      stopifnot(nrow(Xmat_ord) == nrow(Xmat))
      xgb_data <- try(xgboost::xgb.DMatrix(Xmat_ord), silent = TRUE)

      if (self$fit_object$training_offset) {
        offset <- task$offset_transformed(self$fit_object$link_fun, for_prediction = TRUE)
        try(xgboost::setinfo(xgb_data, "base_margin", offset), silent = TRUE)
      }

      ntreelimit <- if (!is.null(fit_object[["best_ntreelimit"]]) &&
                        !("gblinear" %in% fit_object[["params"]][["booster"]])) {
        fit_object[["best_ntreelimit"]]
      } else {
        0
      }

      predictions <- rep.int(list(numeric()), 1)
      if (nrow(Xmat) > 0) {
        predictions <- stats::predict(fit_object, newdata = xgb_data, ntreelimit = ntreelimit, reshape = TRUE)

        if (private$.training_outcome_type$type == "categorical") {
          predictions <- pack_predictions(predictions)
        }
      }

      return(predictions)
    },
    .required_packages = c("xgboost")
  )
)
