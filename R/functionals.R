


# A list of functionals and their corresponding functions for 'fun' (direct estimation)
# and 'rep' (representation). Each element represents a different functional.
functionals_info <- list(
  'ATE' = list(
    # Average Treatment Effect (ATE)
    "fun" = function(mu1, mu0, A, ...) {
      # Directly computes the difference between the treatment and control means.
      return(mu1 - mu0)
    },
    "rep" = function(pi1, pi0, A, Y, weights, ...) {
      # Computes the ATE using a representation formula with propensity scores.
      return(A/pi1 - (1 - A)/pi0)
    }
  ),
  'Y1' = list(
    # Expected outcome under treatment (Y1)
    "fun" = function(mu1, mu0, A, ...) {
      # Returns the expected value of the outcome under treatment.
      return(mu1)
    },
    "rep" = function(pi1, pi0, A, Y, weights, ...) {
      # Computes a representation formula for the treated group.
      return(A/pi1)
    }
  ),
  'Y0' = list(
    # Expected outcome under control (Y0)
    "fun" = function(mu1, mu0, A, ...) {
      # Returns the expected value of the outcome under control.
      return(mu0)
    },
    "rep" = function(pi1, pi0, A, Y, weights, ...) {
      # Computes a representation formula for the control group.
      return((1 - A)/pi0)
    }
  ),
  'ATT' = list(
    # Average Treatment effect on the Treated (ATT)
    "fun" = function(mu1, mu0, A, Y, weights, ...) {
      # Computes the ATT by estimating the mean outcomes among the treated.
      E1Y1 <- weighted.mean(Y[A==1], weights = weights[A==1])
      E1Y0 <- (A/weighted.mean(A, weights = weights)) * mu0
      return(E1Y1 - E1Y0)
    },
    "rep" = function(pi1, pi0, A, Y, weights, ...) {
      # Computes a representation formula for the ATT using propensity scores.
      pA1 <- weighted.mean(A, weights = weights)
      alpha <- A/pA1 - (pi1/pA1) * (1 - A)/pi0
    }
  )
)
