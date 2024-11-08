import xgboost as xgb
import numpy as np

def fit_representer(W, A, weights, m_matrix, params={}, early_stopping_rounds=10, test_size=0.2, refit=True):
    # Prepare data for XGBoost in long format
    unique_A = sorted(np.unique(A))  # Get unique values in A
    long_data = np.vstack(
        [np.column_stack((W, np.full(W.shape[0], a))) for a in unique_A]
    )  # Combine W and corresponding values of A

    # Create labels corresponding to combined data (binary indicator)
    expanded_A = np.hstack([np.where(A == a, 1, 0) for a in unique_A]).reshape(-1)

    # Create weights corresponding to combined data
    expanded_weights = np.hstack([weights for _ in unique_A]).reshape(-1)

    # Create full DMatrix with the combined data
    dtrain = xgb.DMatrix(long_data, label=expanded_A, weight=expanded_weights)

    # Split long data into training and validation sets
    val_split = int((1 - test_size) * long_data.shape[0])
    dtrain_data = xgb.DMatrix(long_data[:val_split], label=expanded_A[:val_split], weight=expanded_weights[:val_split])
    dval_data = xgb.DMatrix(long_data[val_split:], label=expanded_A[val_split:], weight=expanded_weights[val_split:])

    # Define a custom loss and evaluation function
    def make_loss_function(m_matrix):
        def eval_function(preds, data):
            weights = data.get_weight()
            A_indicator = data.get_label().reshape(len(unique_A), -1).T
            weights_short = weights[:A_indicator.shape[0]]
            A = np.where(A_indicator == 1)[1]
            pred_matrix = preds.reshape(len(unique_A), -1).T
            alpha = np.sum(A_indicator * pred_matrix, axis=1)
            m_alpha = np.sum(pred_matrix * m_matrix, axis=1)
            loss = alpha**2 - 2 * m_alpha
            return 'custom_loss', np.average(loss, weights=weights_short)

        def loss_function(preds, data):
            weights = data.get_weight()
            A_indicator = data.get_label()
            grad_alpha = m_matrix.T.reshape(-1)
            gradient = (A_indicator * (preds - grad_alpha)) * weights
            hessian = A_indicator * weights
            return gradient, hessian

        return loss_function, eval_function

    loss_function_train, eval_function_train = make_loss_function(m_matrix[:val_split])

    # Track evaluation results on validation set for early stopping
    evals = [(dtrain_data, 'train'), (dval_data, 'eval')]
    evals_result = {}

    # Train model with custom loss function, evaluation function, and early stopping
    model = xgb.train(
        params,
        dtrain_data,
        obj=loss_function_train,
        feval=eval_function_train,  # Pass custom evaluation function
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False
    )

    # Check if early stopping occurred, refit on the full dataset if refit is enabled
    if refit:
        loss_function_full, eval_function_full = make_loss_function(m_matrix)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=model.best_iteration + 1,
            obj=loss_function_full,
            feval=eval_function_full  # Pass custom evaluation function for refitting
        )

    return model, evals_result

# Generate synthetic data
n = 1000  # Number of rows
W = np.random.uniform(0, 1, size=(n, 2))  # W is uniform, 2 features for example

# Generate A as a Bernoulli indicator based on logistic function of W
logit_probs = 1 / (1 + np.exp(-(W[:, 0] - W[:, 1])))  # Logistic function using W features
A = np.random.binomial(1, logit_probs)  # Bernoulli draws

# Weights (for simplicity, set to 1 here)
weights = np.ones(n)

# m_matrix, as specified, with shape (n, 2)
m_matrix = np.array([[1, -1]] * n)

# Parameters for XGBoost (example parameters)
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror',  # Not used with custom loss
    'verbosity': 0
}

# Call the function
model, evals_result = fit_representer(W, A, weights, m_matrix, params=params, early_stopping_rounds=10, test_size=0.2)

# Print results
print("Model trained.")
print("Evaluation results:", evals_result)
