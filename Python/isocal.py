import xgboost as xgb
import numpy as np

def isoreg_with_xgboost(x, y, max_depth=15, min_child_weight=20):
    """
    Fits isotonic regression using XGBoost with monotonic constraints to ensure 
    non-decreasing predictions as the predictor variable increases.

    Args:
        x (np.array): A vector or matrix of predictor variables.
        y (np.array): A vector of response variables.
        max_depth (int, optional): Maximum depth of the trees in XGBoost. 
                                   Default is 15.
        min_child_weight (float, optional): Minimum sum of instance weights 
                                            needed in a child node. Default is 20.

    Returns:
        function: A prediction function that takes a new predictor variable x 
                  and returns the model's predicted values.
                  
    Example:
        >>> x = np.array([[1], [2], [3]])
        >>> y = np.array([1, 2, 3])
        >>> model = isoreg_with_xgboost(x, y)
        >>> model(np.array([[1.5], [2.5]]))
    """
    
    # Create an XGBoost DMatrix object from the data
    data = xgb.DMatrix(data=np.asarray(x), label=np.asarray(y))

    # Set parameters for the monotonic XGBoost model
    params = {
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'monotone_constraints': "(1)",  # Enforce monotonic increase
        'eta': 1,
        'gamma': 0,
        'lambda': 0
    }

    # Train the model with one boosting round
    iso_fit = xgb.train(params=params, dtrain=data, num_boost_round=1)

    # Prediction function for new data
    def predict_fn(x):
        """
        Predicts output for new input data using the trained isotonic regression model.
        
        Args:
            x (np.array): New predictor variables as a vector or matrix.
        
        Returns:
            np.array: Predicted values.
        """
        data_pred = xgb.DMatrix(data=np.asarray(x))
        pred = iso_fit.predict(data_pred)
        return pred

    return predict_fn
