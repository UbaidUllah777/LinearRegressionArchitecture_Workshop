
"""
Evaluation Module
Linear Regression Architecture Workshop

Provides reusable functions for:
1. Model evaluation (RMSE, MAE, R2)
2. Model comparison reporting
3. Regression line visualization
4. Actual vs. predicted visualization
5. Gradient descent cost visualization
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from datetime import datetime, timezone


from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# --------------------------------------------------
# Project Root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# 1. Model Evaluation
# --------------------------------------------------

def evaluate_model(y_true, y_pred):
    """
    Calculate RMSE, MAE and R2 for model predictions.
    """

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


# --------------------------------------------------
# 2. Evaluation Report
# --------------------------------------------------

def create_evaluation_report(
    y_true,
    scratch_predictions,
    sklearn_predictions
):
    """
    Compare the evaluation metrics of both models.
    """

    scratch_metrics = evaluate_model(
        y_true,
        scratch_predictions
    )

    sklearn_metrics = evaluate_model(
        y_true,
        sklearn_predictions
    )

    report = pd.DataFrame([
        {
            "Model": "From Scratch",
            **scratch_metrics
        },
        {
            "Model": "Scikit-learn",
            **sklearn_metrics
        }
    ])

    return report


# --------------------------------------------------
# Helper: Save a Figure
# --------------------------------------------------

def save_figure(fig, save_path):
    """
    Save a Matplotlib figure.

    Create the destination directory if required.
    """

    path = Path(save_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    return path


# --------------------------------------------------
# 3. Regression Line Comparison
# --------------------------------------------------

def plot_regression_comparison(
    X_original,
    y_true,
    scratch_predictions,
    sklearn_predictions,
    save_path=None
):
    """
    Plot actual values and both regression lines.

    The horizontal axis uses original, unscaled
    median income values.
    """

    x = np.asarray(X_original).ravel()
    y = np.asarray(y_true).ravel()

    scratch_predictions = np.asarray(
        scratch_predictions
    ).ravel()

    sklearn_predictions = np.asarray(
        sklearn_predictions
    ).ravel()

    # Sort input values for drawing regression lines
    sorted_indices = np.argsort(x)

    # Plot a reproducible sample of actual observations
    rng = np.random.default_rng(42)

    sample_indices = rng.choice(
        len(x),
        size=min(1500, len(x)),
        replace=False
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.scatter(
        x[sample_indices],
        y[sample_indices],
        alpha=0.3,
        s=15,
        label="Actual Test Data"
    )

    ax.plot(
        x[sorted_indices],
        scratch_predictions[sorted_indices],
        linewidth=2,
        label="From Scratch"
    )

    ax.plot(
        x[sorted_indices],
        sklearn_predictions[sorted_indices],
        linestyle="--",
        linewidth=2,
        label="Scikit-learn"
    )

    ax.set_title(
        "California Housing - Regression Line Comparison"
    )

    ax.set_xlabel("Median Income (MedInc)")

    ax.set_ylabel(
        "Median House Value (Units of $100,000 USD)"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)

    return fig


# --------------------------------------------------
# 4. Actual vs. Predicted Plot
# --------------------------------------------------

def plot_actual_vs_predicted(
    y_true,
    y_pred,
    save_path=None
):
    """
    Compare actual values against model predictions.
    """

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.3,
        s=15
    )

    # Perfect prediction reference line
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction"
    )

    ax.set_title("Actual vs. Predicted House Values")
    ax.set_xlabel("Actual Median House Value")
    ax.set_ylabel("Predicted Median House Value")

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)

    return fig


# --------------------------------------------------
# 5. Gradient Descent Cost Plot
# --------------------------------------------------

def plot_cost_history(cost_history, save_path=None):
    """
    Plot MSE recorded during gradient descent.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        range(1, len(cost_history) + 1),
        cost_history,
        linewidth=2
    )

    ax.set_title("Gradient Descent - MSE Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Squared Error (MSE)")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)

    return fig




# --------------------------------------------------
# 6. Experiment Tracking
# --------------------------------------------------

def save_experiment_results(report, config):
    """
    Append model evaluation metrics and experiment
    settings to the configured CSV file.
    """

    # Get the configured output location
    results_path = Path(
        config["experiment"]["results_path"]
    )

    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path

    # Create the destination directory if necessary
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Create a copy to avoid modifying the original report
    results = report.copy()

    # Store one shared timestamp for both model results
    run_timestamp = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")

    results.insert(
        0,
        "run_timestamp_utc",
        run_timestamp
    )

    # Record experiment configuration
    results["feature"] = config["model"]["feature"]
    results["target"] = config["model"]["target"]

    results["learning_rate"] = config["model"]["learning_rate"]
    results["iterations"] = config["model"]["iterations"]

    results["test_size"] = config["preprocessing"]["test_size"]
    results["random_state"] = config["preprocessing"]["random_state"]

    results["scaling_method"] = config["preprocessing"]["scaling_method"]

    results["dataset"] = config["data"]["california_csv"]

    # Write headers only if the CSV is new or empty
    write_header = (
        not results_path.exists()
        or results_path.stat().st_size == 0
    )

    # Append the experiment results without overwriting older runs
    results.to_csv(
        results_path,
        mode="a",
        header=write_header,
        index=False
    )

    return results_path






# --------------------------------------------------
# Direct Execution - Configuration-Driven Experiment
# --------------------------------------------------

if __name__ == "__main__":

    from src.data_loader import load_config, load_csv
    from src.preprocessing import prepare_data

    from src.model import (
        gradient_descent,
        predict_from_scratch,
        train_sklearn_model
    )

    print("Starting Configuration-Driven Evaluation...")

    # 1. Read settings from YAML
    config = load_config()

    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    model_config = config["model"]
    experiment_config = config["experiment"]

    # 2. Load the configured dataset
    df = load_csv(
        data_config["california_csv"]
    )

    # 3. Preprocess using configured parameters
    prepared = prepare_data(
        df=df,
        feature=model_config["feature"],
        target=model_config["target"],
        test_size=preprocessing_config["test_size"],
        random_state=preprocessing_config["random_state"],
        scaling_method=preprocessing_config["scaling_method"]
    )

    X_train_scaled = prepared["X_train_scaled"]
    X_test_scaled = prepared["X_test_scaled"]

    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    # 4. Train the from-scratch implementation
    theta_0, theta_1, cost_history = gradient_descent(
        x=X_train_scaled,
        y=y_train,
        learning_rate=model_config["learning_rate"],
        iterations=model_config["iterations"]
    )

    # 5. Train the scikit-learn implementation
    sk_model = train_sklearn_model(
        X_train_scaled,
        y_train
    )

    # 6. Generate testing predictions
    scratch_predictions = predict_from_scratch(
        X_test_scaled,
        theta_0,
        theta_1
    )

    sklearn_predictions = sk_model.predict(
        X_test_scaled
    )

    # 7. Evaluate both models
    report = create_evaluation_report(
        y_test,
        scratch_predictions,
        sklearn_predictions
    )

    print("\nExperiment Evaluation Results:")
    print(report.round(6).to_string(index=False))

    # 8. Save experiment metrics to CSV
    results_path = save_experiment_results(
        report,
        config
    )

    print("\nExperiment results saved to:")
    print(results_path.relative_to(PROJECT_ROOT))

    # 9. Read configured figure directory
    figures_dir = Path(
        experiment_config["figures_dir"]
    )

    # 10. Save regression comparison
    fig1 = plot_regression_comparison(
        prepared["X_test"][model_config["feature"]],
        y_test,
        scratch_predictions,
        sklearn_predictions,
        save_path=figures_dir / "regression_comparison.png"
    )

    plt.close(fig1)

    # 11. Save actual vs. predicted plot
    fig2 = plot_actual_vs_predicted(
        y_test,
        sklearn_predictions,
        save_path=figures_dir / "actual_vs_predicted.png"
    )

    plt.close(fig2)

    # 12. Save gradient descent convergence plot
    fig3 = plot_cost_history(
        cost_history,
        save_path=figures_dir / "cost_history.png"
    )

    plt.close(fig3)

    print("\nThree evaluation figures saved successfully.")

    print(
        "\nBoth models produce approximately equal predictions:",
        np.allclose(
            scratch_predictions,
            sklearn_predictions,
            atol=1e-4
        )
    )

    print("\nExperiment completed successfully!")
