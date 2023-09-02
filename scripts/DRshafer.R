library(data.table)
library(sl3)
library(xgboost)
d <- 3
seed_start <- 98103
set.seed(seed_start)

do_sims <- function(n, nsims) {
  pos_const <- 2 #not used

  cols <- paste0("W", 1:d)
  stack_parametric <- Lrnr_earth$new(degree=2, nk = 100, pmethod = "cv", nfold = 3)
    # list(
    #   Lrnr_earth$new(degree=1, nk = 100, pmethod = "cv", nfold = 3),
    #   Lrnr_earth$new(degree=2, nk = 100, pmethod = "cv", nfold = 3),
    #   Lrnr_earth$new(degree=3, nk = 100, pmethod = "cv", nfold = 3),
    #   Lrnr_gam$new(),
    #   Lrnr_glm$new(),
    #   Lrnr_glmnet$new(formula = ~.^2)
    # )



  stack_rf <-   Lrnr_ranger$new(max.depth = 12)

  stack_xg <- #Stack$new(
    list(
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 2, nrounds = 20, eta = 0.3 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 3, nrounds = 20, eta = 0.3 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 4, nrounds = 20, eta = 0.3 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 5, nrounds = 20, eta = 0.3 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 6, nrounds = 20, eta = 0.3 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 2, nrounds = 20, eta = 0.25 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 3, nrounds = 20, eta = 0.25 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 4, nrounds = 20, eta = 0.25 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 5, nrounds = 20, eta = 0.25 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 6, nrounds = 20, eta = 0.25 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 2, nrounds = 20, eta = 0.2 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 3, nrounds = 20, eta = 0.2 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 4, nrounds = 20, eta = 0.2 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 5, nrounds = 20, eta = 0.2 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 6, nrounds = 20, eta = 0.2 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 2, nrounds = 20, eta = 0.15 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 3, nrounds = 20, eta = 0.15 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 4, nrounds = 20, eta = 0.1 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 5, nrounds = 20, eta = 0.1 ),
      Lrnr_xgboost$new(min_child_weight = 5, max_depth = 6, nrounds = 20, eta = 0.1 )
    )
 # )





  metalearner <-  Lrnr_cv_selector$new(loss_squared_error)
  lrnr_parametric <-  Pipeline$new(Lrnr_cv$new(stack_parametric), metalearner)
  lrnr_rf <-   Pipeline$new(Lrnr_cv$new(stack_rf),Lrnr_cv_selector$new(loss_squared_error))
  lrnr_xg <-  Pipeline$new(Lrnr_cv$new(stack_xg), metalearner)


  sim_results <- lapply(1:nsims, function(i){
    try({
      set.seed(seed_start + i)
      print(paste0("iter: ", i))
      data_list <- get_data(n, pos_const)
      W <- data_list$W
      A <- data_list$A
      Y <- data_list$Y
      ATE <- data_list$ATE
      n <- length(A)
      folds <- 5
      initial_estimators_parametric <- compute_initial(W,A,Y, lrnr_mu = lrnr_parametric, lrnr_pi = lrnr_parametric, folds = 5, invert = FALSE)
      folds <- initial_estimators_parametric$folds
      initial_estimators_xg <- compute_initial(W,A,Y, lrnr_mu = lrnr_xg, lrnr_pi = lrnr_xg, folds = folds, invert = FALSE)
      initial_estimators_rf <- compute_initial(W,A,Y, lrnr_mu = lrnr_rf, lrnr_pi = lrnr_rf, folds = folds, invert = FALSE)
      initial_estimators_misp <- compute_initial(W,A,Y, lrnr_mu =  Lrnr_cv$new(Lrnr_glm$new()), lrnr_pi =  Lrnr_cv$new(Lrnr_glm$new()), folds = folds)

      out_list <- list()

      for(lrnr in c("parametric", "xg",   "rf" )) {
        initial_estimators <- get(paste0("initial_estimators_", lrnr))
        for(misp in c("1", "2", "3" )) {
          mu1 <- initial_estimators$mu1
          mu0 <- initial_estimators$mu0
          pi1 <- initial_estimators$pi1
          pi0 <- initial_estimators$pi0
          if(misp == "2") {
            mu1 <- initial_estimators_misp$mu1
            mu0 <- initial_estimators_misp$mu0
          } else if(misp == "3" ) {
            pi1 <- initial_estimators_misp$pi1
            pi0 <- initial_estimators_misp$pi0
          } else if(misp == "4") {
            mu1 <- initial_estimators_misp$mu1
            mu0 <- initial_estimators_misp$mu0
            pi1 <- initial_estimators_misp$pi1
            pi0 <- initial_estimators_misp$pi0
          }


          out_AIPW <- compute_AIPW(A,Y, mu1=mu1, mu0 =mu0, pi1 = pi1, pi0 = pi0)
          out_AuDRIE <- compute_AuDRIE_boot(A,Y,  mu1=mu1, mu0 =mu0, pi1 = pi1, pi0 = pi0, nboot = 500, folds = folds, alpha = 0.05)
          out <- as.data.table(rbind(unlist(out_AuDRIE), unlist(out_AIPW)))
          colnames(out) <- c("estimate", "CI_left", "CI_right")
          out$misp <- misp
          out$estimator <- c("auDRI", "AIPW")
          out$lrnr <- lrnr
          out_list[[paste0(misp, lrnr)]] <- out
        }
      }
      out <- rbindlist(out_list)
      out$pos_const <- pos_const
      out$n <- n
      out$ATE <- ATE
      out$iter <- i
      return(as.data.table(out))
    })
    return(data.table())
  })
  sim_results <- data.table::rbindlist(sim_results)
  key <- paste0("DR_iter=", nsims, "_n=", n, "_pos=", pos_const )
  try({fwrite(sim_results, paste0("~/DRinference/simResultsDR/sim_results_", key, ".csv"))})
  return(sim_results)
}


get_data <- function(n, pos_const) {
  # setting of Kang and Shafer (2008): https://projecteuclid.org/journals/statistical-science/volume-22/issue-4/Demystifying-Double-Robustness--A-Comparison-of-Alternative-Strategies-for/10.1214/07-STS227.full
  # Z <- replicate(d, rnorm(n))
  # m <- 5 + 5 * Z[,1] + 2 * Z[,2] + 2 * Z[,3] + 2 * Z[,4]
  # tau <- 1 - 2 * Z[,1] + Z[,2] + Z[,3] - Z[,4]
  # pi = plogis(-Z[,1] + 0.5*Z[,2] - 0.25*Z[,3] - 0.1*Z[,4])
  # A <- rbinom(n, 1, pi)
  # mu0 <- m + (0 - pi) * tau
  # mu1 <- m + (1 - pi) * tau
  # Y <- A * mu1 + (1-A) * mu0 + rnorm(n, 0, 2)

  W <- replicate(d, runif(n, -1, 1))
  colnames(W) <- paste0("W", 1:d)
  link <- (sign(W[,1]) * sqrt(abs(W[,1])) + sin(3.14*W[,2]) + W[,3]*sin(W[,3]) - 0.5)
  pi <- plogis(1.5 * link)
  A <- rbinom(n, 1, pi)
  mu0 <-  plogis(-1 + cos(3.14*W[,1]) + W[,2]* sin(W[,2]) + sign(W[,3]) * sqrt(abs(W[,3])))
  mu1 <- plogis(qlogis(mu0) + 1 +  (sign(W[,1]) * sqrt(abs(W[,1])) + sin(3.14*W[,2]) + W[,3]*sin(W[,3])))
  mu <- ifelse(A==1, mu1, mu0)
  Y <- rbinom(n, 1, mu)


  W <- cbind(exp(W[,1]),
             W[,1] * W[,2],
             W[,1] * W[,3]
  )



  # W <- cbind(exp(Z[,1]/2),
  #           1 + Z[,2] / (1 + exp(Z[,1])),
  #            Z[,1] * Z[,3],
  #            (Z[,2] + Z[,4] + 1)^2
  # )


  colnames(W) <- paste0("W", 1:d)
  out <- list(W = W, A = A, Y = Y,  pi = pi, mu0 = mu0, mu1 = mu1, ATE = 0.2462516)
  return(out)
}




compute_AuDRIE_boot <-  function(A,Y, mu1, mu0, pi1, pi0, nboot = 1000, folds, alpha = 0.05) {
  data <- data.table(A, Y, mu1, mu0, pi1, pi0)
  folds <- lapply(folds, `[[`, "validation_set")
  tau_n <- compute_AuDRIE(A,Y, mu1, mu0, pi1, pi0)

  bootstrap_estimates <- sapply(1:nboot, function(iter){
    try({
      bootstrap_indices <- unlist(lapply(folds, function(fold) {
        sample(fold, length(fold), replace = TRUE)
      }))
      data_boot <- data[bootstrap_indices,]
      tau_boot <- compute_AuDRIE(data_boot$A, data_boot$Y, mu1 = data_boot$mu1, mu0 = data_boot$mu0, pi1 = data_boot$pi1, pi0 =data_boot$pi0)
      return(tau_boot)
    })
    return(NULL)
  })




  CI <- tau_n + quantile(bootstrap_estimates - median(bootstrap_estimates), c(alpha/2, 1-alpha/2), na.rm = TRUE)
  return(list(estimate = tau_n, CI = CI))
}



compute_AuDRIE <- function(A,Y, mu1, mu0, pi1, pi0) {
  calibrated_estimators  <- calibrate_nuisances(A,Y, mu1 = mu1, mu0 = mu0, pi1 = pi1, pi0 = pi0)
  mu1_star <- calibrated_estimators$mu1_star
  mu0_star <- calibrated_estimators$mu0_star
  pi1_star <- calibrated_estimators$pi1_star
  pi0_star <- calibrated_estimators$pi0_star

  mu_star <- ifelse(A==1, mu1_star, mu0_star)
  alpha_n <- ifelse(A==1, 1/pi1_star, - 1/pi0_star)
  tau_n <-  mean(mu1_star - mu0_star + alpha_n * (Y - mu_star))
  return(tau_n)
}

compute_AIPW <- function(A,Y, mu1, mu0, pi1, pi0) {
  n <- length(A)
  pi1 <- pmax(pi1,  25/(sqrt(n)*log(n)))
  pi0 <- pmax(pi0,  25/(sqrt(n)*log(n)))
  mu <- ifelse(A==1, mu1, mu0)
  alpha_n <- ifelse(A==1, 1/pi1, - 1/pi0)
  tau_n <-  mean(mu1 - mu0 + alpha_n * (Y - mu))
  CI <- tau_n + c(-1,1) * qnorm(1-0.025) * sd(mu1 - mu0 + alpha_n * (Y - mu))/sqrt(n)
  return(list(estimate = tau_n, CI = CI))
}


isoreg_with_xgboost <- function(x,y,max_depth = 15, min_child_weight = 10) {
  data <- xgboost::xgb.DMatrix(data = as.matrix(x), label = as.vector(y))
  iso_fit <- xgboost::xgb.train(params = list(max_depth = max_depth,
                                              min_child_weight = min_child_weight,
                                              monotone_constraints = 1,
                                              eta = 1, gamma = 0,
                                              lambda = 0),
                                data = data, nrounds = 1)
  fun <- function(x) {
    data_pred <- xgboost::xgb.DMatrix(data = as.matrix(x))
    pred <- predict(iso_fit, data_pred)
    return(pred)
  }
  return(fun)
}

calibrate_nuisances <- function(A, Y,mu1, mu0, pi1, pi0) {

  calibrator_mu1 <- isoreg_with_xgboost(mu1[A==1], Y[A==1])
  mu1_star <- calibrator_mu1(mu1)
  calibrator_mu0 <- isoreg_with_xgboost(mu0[A==0], Y[A==0])
  mu0_star <- calibrator_mu0(mu0)


  calibrator_pi1 <- isoreg_with_xgboost(pi1, A)
  pi1_star <- calibrator_pi1(pi1)
  calibrator_pi0 <- isoreg_with_xgboost(pi0, 1-A)
  pi0_star <- calibrator_pi0(pi0)
  return(list(mu1_star= mu1_star, mu0_star=mu0_star, pi1_star = pi1_star, pi0_star = pi0_star))
}




compute_initial <- function(W,A,Y, lrnr_mu, lrnr_pi, folds,   invert = FALSE) {
  data <- data.table(W,A,Y)
  print(lrnr_mu)
  taskY0 <- sl3_Task$new(data, covariates = colnames(W), outcome  = "Y",   folds = folds)
  folds <- taskY0$folds
  fit0 <- lrnr_mu$train(taskY0[A==0])
  mu0 <- as.vector(fit0$predict(taskY0))
  data$mu0 <- qlogis(mu0)
  taskY1 <- sl3_Task$new(data, covariates = c(colnames(W)), outcome  = "Y",   folds = folds)
  fit1 <- lrnr_mu$train(taskY1[A==1])
  mu1 <-  as.vector(fit1$predict(taskY1))
  print(fit0$fit_object$learner_fits$Lrnr_cv_selector_NULL$fit_object$cv_risk)
  print(fit1$fit_object$learner_fits$Lrnr_cv_selector_NULL$fit_object$cv_risk)


  taskA <- sl3_Task$new(data, covariates = colnames(W), outcome  = "A", folds = folds )

  fit1 <- lrnr_pi$train(taskA)
  pi1 <- as.vector(fit1$predict(taskA))
  pi0 <- 1- pi1


  print("done")
  return(list(mu1 = mu1, mu0 = mu0, pi1 = pi1, pi0 = pi0, folds = folds))
}




