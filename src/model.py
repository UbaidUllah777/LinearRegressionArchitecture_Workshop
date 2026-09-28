
"""
Model Module
Linear Regression Architecture Workshop

Contains:
1. Linear regression hypothesis function
2. Mean Squared Error cost function
3. Gradient descent implementation from scratch
4. Scikit-learn linear regression implementation
"""

import numpy as np

from sklearn.linear_model import LinearRegression


# --------------------------------------------------
# 1. Hypothesis Function
# --------------------------------------------------

def predict_from_scratch(x, theta_0, theta_1):
    """
    Generate predictions using:

    h(x) = theta_0 + theta_1 * x
    """

    x = np.asarray(x).ravel()

    return theta_0 + theta_1 * x


# --------------------------------------------------
# 2. Mean Squared Error
# --------------------------------------------------

def mse_cost(y_true, y_pred):
    """
    Calculate Mean Squared Error from scratch.
    """

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    return np.mean((y_true - y_pred) ** 2)


# --------------------------------------------------
# 3. Gradient Descent
# --------------------------------------------------

def gradient_descent(
    x,
    y,
    learning_rate=0.1,
    iterations=1000
):
    """
    Train univariate linear regression from scratch
    using gradient descent.

    Returns:
        theta_0: Learned intercept
        theta_1: Learned slope
        cost_history: MSE recorded after each iteration
    """

    # Convert input data into 1D NumPy arrays
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if len(x) != len(y) or len(y) == 0:
        raise ValueError(
            "X and y must have the same non-zero length."
        )

    if learning_rate <= 0 or iterations < 1:
        raise ValueError(
            "Learning rate and iterations must be positive."
        )

    # Initialize model parameters
    theta_0 = 0.0
    theta_1 = 0.0

    # Number of training observations
    m = len(y)

    # Record the cost after every iteration
    cost_history = []

    for _ in range(iterations):

        # Step 1: Generate predictions
        y_pred = predict_from_scratch(
            x,
            theta_0,
            theta_1
        )

        # Step 2: Calculate prediction errors
        errors = y_pred - y

        # Step 3: Calculate gradients
        gradient_0 = (2 / m) * np.sum(errors)

        gradient_1 = (2 / m) * np.sum(
            errors * x
        )

        # Step 4: Update parameters
        theta_0 -= learning_rate * gradient_0

        theta_1 -= learning_rate * gradient_1

        # Step 5: Calculate updated predictions
        updated_predictions = predict_from_scratch(
            x,
            theta_0,
            theta_1
        )

        # Step 6: Calculate and record MSE
        cost = mse_cost(
            y,
            updated_predictions
        )

        cost_history.append(cost)

    return theta_0, theta_1, cost_history


# --------------------------------------------------
# 4. Scikit-learn Implementation
# --------------------------------------------------

def train_sklearn_model(X_train, y_train):
    """
    Train scikit-learn's LinearRegression model.
    """

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model




# --------------------------------------------------
# Direct Execution - Configuration Driven
# --------------------------------------------------

if __name__ == "__main__":

    from src.data_loader import load_config, load_csv
    from src.preprocessing import prepare_data

    print("Testing Model Module...")

    # 1. Load experiment configuration
    config = load_config()

    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    model_config = config["model"]

    # 2. Load the configured dataset
    df = load_csv(
        data_config["california_csv"]
    )

    # 3. Preprocess using YAML settings
    prepared = prepare_data(
        df=df,
        feature=model_config["feature"],
        target=model_config["target"],
        test_size=preprocessing_config["test_size"],
        random_state=preprocessing_config["random_state"],
        scaling_method=preprocessing_config["scaling_method"]
    )

    X_train_scaled = prepared["X_train_scaled"]
    y_train = prepared["y_train"]

    # 4. Train from-scratch model using YAML settings
    theta_0, theta_1, cost_history = gradient_descent(
        x=X_train_scaled,
        y=y_train,
        learning_rate=model_config["learning_rate"],
        iterations=model_config["iterations"]
    )

    print("\nConfigured Experiment Settings:")

    print("Feature:", model_config["feature"])
    print("Target:", model_config["target"])
    print("Learning rate:", model_config["learning_rate"])
    print("Iterations:", model_config["iterations"])

    print("\nFrom-Scratch Model:")
    print(f"Intercept: {theta_0:.6f}")
    print(f"Slope: {theta_1:.6f}")
    print(f"Final MSE: {cost_history[-1]:.6f}")

    # 5. Train scikit-learn using the same training data
    sk_model = train_sklearn_model(
        X_train_scaled,
        y_train
    )

    print("\nScikit-learn Model:")
    print(f"Intercept: {sk_model.intercept_:.6f}")
    print(f"Slope: {sk_model.coef_[0]:.6f}")

    # 6. Compare both implementations
    parameters_match = (
        np.isclose(
            theta_0,
            sk_model.intercept_,
            atol=1e-4
        )
        and
        np.isclose(
            theta_1,
            sk_model.coef_[0],
            atol=1e-4
        )
    )

    print(
        "\nModel parameters match:",
        parameters_match
    )

