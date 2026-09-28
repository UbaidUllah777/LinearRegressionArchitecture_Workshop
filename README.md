
# Linear Regression Architecture Workshop

## 1. Project Overview

This project demonstrates univariate linear regression and modular MLOps architecture using real housing data from California (USA) and Ontario (Canada).

It was developed as part of the Foundations of Machine Learning workshop.

The primary objective is to predict California's median house value using median income as a single predictor and compare two implementations:

1. Linear Regression from Scratch using Gradient Descent.
2. Linear Regression using Scikit-learn.

The project also demonstrates:

- Data collection from CSV files, a web API, and PostgreSQL.
- Data cleaning and exploratory data analysis (EDA).
- Feature standardization and train/test splitting.
- Model training and evaluation.
- Modular Python architecture.
- YAML-based experiment configuration.
- Experiment tracking and reproducibility.

---

## 2. Data Sources

### 2.1 California Housing Dataset

**Source:** Scikit-learn California Housing Dataset

https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

- Total observations: 20,640
- Total columns: 9
- Predictor (X): `MedInc` — Median Income
- Target (y): `MedHouseVal` — Median House Value
- Missing values: 0
- Purpose: Primary dataset for univariate linear regression.

`MedHouseVal` represents historical median house values in units of $100,000 USD.

For example, a value of `2.5` represents approximately $250,000.

The dataset is historical and contains capped target values at approximately $500,000. It does not represent current California property listings or individual house sale prices.

### 2.2 Ontario Housing Dataset

**Source:** Statistics Canada, Table 18-10-0205-01 — New housing price index, monthly.

https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810020501

The dataset was retrieved through the Statistics Canada Web Data Service API using Python's `requests` library.

- Geography: Ontario
- Selected category: Total (house and land)
- Original Ontario records: 1,644
- Selected category records: 548
- Records removed due to unavailable index values: 60
- Final processed observations: 488
- Processed period: January 1986 – August 2026
- Unit: Index, December 2016 = 100

The Ontario dataset contains a housing price index, not individual property sale prices.

It is used to demonstrate API data collection, preprocessing, exploratory analysis, and relational database integration.

**Note:** The included CSV represents the dataset used in this experiment. Re-downloading it from the live Statistics Canada API may return additional observations when new monthly data becomes available.

---

## 3. Technology Stack

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SQLAlchemy
- Psycopg 3
- Requests
- PyYAML
- Python-dotenv
- Jupyter Notebook
- Neon PostgreSQL
- Visual Studio Code

All Python dependencies and their installed versions are recorded in `requirements.txt`.

---

## 4. Project Structure

```text
LinearRegressionArchitecture_Workshop/
│
├── data/
│   ├── raw/
│   │   ├── california_housing.csv
│   │   ├── statcan_nhpi_1810020501.zip
│   │   └── statcan_nhpi/
│   │       ├── 18100205.csv
│   │       └── 18100205_MetaData.csv
│   │
│   └── processed/
│       └── ontario_housing_index.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── linear_regression.ipynb
│   ├── RobotPM_MLOps.ipynb
│   └── images/
│       ├── PM_ProcessFlow.png
│       └── RobotPM_MLOps.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
│
├── configs/
│   └── experiment_config.yaml
│
├── experiments/
│   ├── results.csv
│   └── figures/
│       ├── regression_comparison.png
│       ├── actual_vs_predicted.png
│       └── cost_history.png
│
├── .env                 # Local only; excluded from Git
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 5. Environment Setup

Follow these instructions to install and execute the project on a new computer.

### Step 1 — Prerequisites

Ensure the following applications are installed:

- Python 3.10 or newer.
- Git.
- Visual Studio Code (recommended for notebooks).
- Python and Jupyter extensions for VS Code.

A PostgreSQL database is optional for the main regression experiment but required to reproduce the database-related activities.

### Step 2 — Clone the GitHub Repository

Open a terminal and execute:

```bash
git clone https://github.com/UbaidUllah777/LinearRegressionArchitecture_Workshop.git

cd LinearRegressionArchitecture_Workshop
```

All subsequent Python module commands should be executed from this main project directory unless otherwise stated.

### Step 3 — Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment.

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

After activation, the terminal should display `(.venv)`.

### Step 4 — Install Project Dependencies

Execute:

```bash
python -m pip install -r requirements.txt
```

This installs the dependencies required by the project, including the PostgreSQL driver and Jupyter packages.

### Step 5 — Verify the Python Environment

Run:

```bash
python -c "import numpy, pandas, sklearn, matplotlib, sqlalchemy, requests, yaml, psycopg; print('Dependencies installed successfully')"
```

Expected output:

```text
Dependencies installed successfully
```

---

## 6. PostgreSQL Configuration

This project uses Neon PostgreSQL for its relational database activities.

**PostgreSQL is not required for the main California regression experiment**, which reads the included CSV.

However, a database connection is required to reproduce the PostgreSQL operations in `EDA.ipynb` and to test the `load_postgres()` function.

### Step 1 — Create a PostgreSQL Database

Visit:

https://neon.tech/

Create a project and obtain its PostgreSQL connection string.

### Step 2 — Create the Environment File

Inside the main project directory, create a file named:

`.env`

Add the following variable:

`Please Note that i will send the env file as zip with no password on it, please download it , unzip and place it in root folder`

```dotenv
DATABASE_URL=neon_postgresql_connection_string
```

Replace the placeholder with your actual database connection string.

The current project uses the Psycopg 3 PostgreSQL driver. Use a compatible connection URL supplied by your database provider.

**Security:** The `.env` file is excluded from Git. Never commit database credentials to GitHub or place them directly in notebooks, the README, or the YAML configuration.

### Step 3 — Populate the Database

Open `notebooks/EDA.ipynb` and execute the data-loading and PostgreSQL sections in order.

The notebook creates and populates the following database tables:

- `california_housing`
- `ontario_housing`

It also demonstrates data retrieval and SQL filtering.

**Important:** The notebook uses `if_exists="replace"` when uploading these tables. Therefore, run its database-loading cells against your own database or a database where replacing these tables is acceptable.

After populating the tables, the SQL retrieval examples and PostgreSQL loader can be executed.

---

## 7. How to Run the Project

The project provides four independent and directly executable Python modules.

Open a terminal in the project root and ensure the virtual environment is activated.

### 7.1 Data Loading

```bash
python -m src.data_loader
```

Tests loading the California Housing CSV.

The module also provides reusable functions for:

- Local CSV loading.
- Statistics Canada API retrieval.
- PostgreSQL queries.
- YAML configuration loading.

API retrieval requires internet access, while database retrieval requires the PostgreSQL configuration described in Section 6.

### 7.2 Data Preprocessing

```bash
python -m src.preprocessing
```

Performs:

- Required-column validation.
- Missing and invalid value handling.
- Predictor and target selection.
- Training and testing split.
- Feature standardization.

The scaler is fitted using training data only to prevent data leakage.

### 7.3 Model Training

```bash
python -m src.model
```

Trains two regression implementations:

1. From-scratch gradient descent.
2. Scikit-learn LinearRegression.

The experiment parameters are read from `configs/experiment_config.yaml`.

The output includes the learned intercept, slope, training MSE, and a comparison of both implementations.

### 7.4 Complete Experiment and Evaluation

To execute the complete configuration-driven regression experiment, run:

```bash
python -m src.evaluation
```

This is the primary execution command for the completed modular project.

It automatically:

1. Loads the experiment configuration.
2. Reads the California housing dataset.
3. Cleans and preprocesses the selected variables.
4. Splits the observations into training and testing subsets.
5. Standardizes the input feature.
6. Trains the gradient descent implementation.
7. Trains the scikit-learn implementation.
8. Generates predictions using the testing dataset.
9. Calculates RMSE, MAE, and R².
10. Records experiment results in `experiments/results.csv`.
11. Saves model evaluation visualizations.

The terminal should display:

```text
Both models produce approximately equal predictions: True

Experiment completed successfully!
```

This command uses the included California CSV and does not require a PostgreSQL connection.

---

## 8. Running the Jupyter Notebooks

The project contains three Jupyter notebooks.

| Notebook | Purpose |
|---|---|
| `EDA.ipynb` | California and Ontario data sourcing and exploratory data analysis |
| `linear_regression.ipynb` | Regression from scratch, scikit-learn comparison, metrics, and visualizations |
| `RobotPM_MLOps.ipynb` | Architecture documentation and future MLOps extensions |

### Running in Visual Studio Code

1. Open the cloned project directory in VS Code.
2. Install the Python and Jupyter extensions, if required.
3. Activate the project's virtual environment.
4. Open the desired notebook.
5. Click **Select Kernel**.
6. Select the Python interpreter from the project's `.venv`.
7. Choose **Restart Kernel and Run All** to execute the notebook from the beginning.

Run the computational notebooks in this order:

1. `notebooks/EDA.ipynb`
2. `notebooks/linear_regression.ipynb`

`RobotPM_MLOps.ipynb` primarily contains Markdown documentation, architectural explanations, and proposed implementation stubs.

### Important Notebook Working Directory

The data-processing notebooks use relative paths such as:

```python
../data/raw/california_housing.csv
```

Therefore, their working directory should be the project's `notebooks/` directory.

To check the current working directory, execute the following inside a notebook:

```python
from pathlib import Path
print(Path.cwd())
```

If using Jupyter Notebook directly, you can start it from the notebooks directory:

```bash
cd notebooks
python -m jupyter notebook
```

The EDA notebook includes external API downloads and PostgreSQL operations. These require internet access and, for database-related cells, a valid PostgreSQL configuration.

The standalone California regression notebook and the modular regression experiment use the included California CSV.

---

## 9. Experiment Configuration

All experiment settings are stored in:

`configs/experiment_config.yaml`

The configuration defines:

- California and Ontario dataset paths.
- Statistics Canada API endpoint.
- PostgreSQL environment variable and table names.
- Selected predictor and target.
- Learning rate and number of iterations.
- Train/test split and random state.
- Feature scaling method.
- Experiment output locations.

### Current Experiment Settings

| Parameter | Value |
|---|---|
| Dataset | California Housing Dataset |
| Predictor | `MedInc` |
| Target | `MedHouseVal` |
| Training data | 80% |
| Testing data | 20% |
| Random state | 42 |
| Scaling | StandardScaler |
| Learning rate | 0.1 |
| Iterations | 1,000 |

The configuration-driven workflow is implemented and verified.

Experiment settings can be modified in the YAML file without changing the underlying regression algorithms.

Database credentials are stored separately in `.env`.

---

## 10. Model Evaluation Results

Both models were evaluated using the same 4,128 testing observations.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| From Scratch | 0.842090 | 0.629909 | 0.458859 |
| Scikit-learn | 0.842090 | 0.629909 | 0.458859 |

Both implementations produced matching results to six decimal places.

The R² result indicates that median income alone explains approximately 45.9% of the variation in median house values in the testing dataset.

RMSE and MAE are expressed in the dataset's target units of $100,000 USD.

### Gradient Descent Results

| Parameter | Result |
|---|---:|
| Intercept | 2.071947 |
| Slope | 0.798520 |
| Initial training MSE | 3.854727 |
| Final training MSE | 0.699145 |
| Iterations | 1,000 |

The decreasing MSE curve demonstrates that gradient descent successfully converged.

---

## 11. Expected Outputs

After executing:

```bash
python -m src.evaluation
```

The following outputs are generated:

| Output | Location |
|---|---|
| Evaluation metrics | Terminal |
| Experiment tracking | `experiments/results.csv` |
| Regression line comparison | `experiments/figures/regression_comparison.png` |
| Actual vs. predicted values | `experiments/figures/actual_vs_predicted.png` |
| Gradient descent convergence | `experiments/figures/cost_history.png` |

Each execution appends two experiment records:

- From Scratch
- Scikit-learn

Existing records are preserved.

The evaluation figures are saved to the configured figures directory. Subsequent executions update the figure files at those output paths.

---

## 12. Reproducing the Results

To reproduce the original regression results:

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Keep the original California CSV and YAML settings unchanged.
5. Execute the complete experiment:

```bash
python -m src.evaluation
```

Both models use identical training and testing observations.

The fixed random state ensures that repeated experiments use the same train/test split.

Feature standardization is fitted on training data only.

Repeated executions using the same dataset, configuration, and environment should reproduce the same evaluation metrics.

The results can be inspected in `experiments/results.csv`.

**Note:** The CSV uses append mode, so repeated experiments increase the number of tracked records. This is expected behaviour.

---

## 13. Key Design Decisions

The following decisions were made to align the project with the workshop requirements:

- Selected median income as the single predictor to satisfy the univariate regression requirement.
- Used the California Housing Dataset as the primary regression dataset.
- Used Statistics Canada's official housing price index API to demonstrate Ontario data collection.
- Standardized the predictor using training data only to prevent data leakage.
- Implemented gradient descent from scratch using NumPy.
- Compared its output with scikit-learn using identical training and testing observations.
- Separated data loading, preprocessing, modelling, and evaluation into independent Python modules.
- Used relative paths to improve project portability.
- Stored experiment parameters in YAML.
- Kept PostgreSQL credentials outside source control using `.env`.
- Stored experiment results in a persistent CSV file.
- Used a fixed random state to support reproducibility.

---

## 14. MLOps Architecture and Reproducibility

### 14.1 Separation of Concerns

The project uses four independent Python modules.

| Module | Responsibility |
|---|---|
| `src/data_loader.py` | CSV, API, PostgreSQL, and YAML configuration loading |
| `src/preprocessing.py` | Cleaning, feature selection, splitting, and standardization |
| `src/model.py` | From-scratch and scikit-learn regression |
| `src/evaluation.py` | Evaluation, reporting, visualization, and experiment tracking |

This design reduces code duplication and makes the components easier to maintain, test, and reuse.

Each module is directly executable and import-safe.

### 14.2 Configuration Management

The project reads experiment settings from `configs/experiment_config.yaml`.

This separates experiment parameters from the underlying implementation.

The PostgreSQL connection URL is stored separately in `.env`.

### 14.3 Experiment Tracking

Experiment results are saved to:

`experiments/results.csv`

Each tracked record includes:

- UTC execution timestamp.
- Model name.
- RMSE, MAE, and R².
- Dataset path.
- Selected predictor and target.
- Learning rate and iterations.
- Train/test split.
- Random state.
- Scaling method.

Previous experiments are preserved for comparison.

### 14.4 Reproducibility

Reproducibility is supported through:

- Fixed random-state configuration.
- Identical datasets and data splits for both implementations.
- Training-only scaler fitting.
- Versioned Python dependencies in `requirements.txt`.
- Centralized YAML experiment parameters.
- Persisted evaluation metrics.

Repeated executions using the same configuration were tested and produced matching evaluation results.

### 14.5 Real-World Pipeline Alignment

The project follows this machine learning workflow:

```text
Data Loading
     |
     v
Preprocessing
     |
     v
Model Training
     |
     v
Model Evaluation
     |
     v
Experiment Tracking
```

The modular structure supports future improvements, including automated testing, model versioning, structured logging, CI/CD, and deployment.

---

## 15. Robot PM MLOps — Future Architectural Extensions

The updated `notebooks/RobotPM_MLOps.ipynb` preserves the professor's original architectural reference and documents its application to this workshop.

It includes:

- Current project architecture and module responsibilities.
- Recommended Additions mapped to the housing regression project.
- Recommended Enhancements and design patterns.
- DataExtractionAnalysis Breakdown examples.
- A proposed class stub for future implementation.

Advanced capabilities such as automated CI/CD, model registries, data drift detection, and structured monitoring are documented as future enhancements.

These features are not presented as completed implementations.

---

## 16. Troubleshooting

### ModuleNotFoundError: No module named 'src'

Make sure the terminal is located in the main project directory before executing module commands.

For example:

```bash
python -m src.evaluation
```

Run this from the repository root, not from inside `src/`.

### Missing Python Packages

Ensure the virtual environment is activated, then run:

```bash
python -m pip install -r requirements.txt
```

### PostgreSQL Connection Error

Verify that:

- A valid `.env` file exists in the project root.
- `DATABASE_URL` contains the correct PostgreSQL connection string.
- The database is accessible.
- The PostgreSQL driver has been installed.

If your connection URL uses the `postgresql+psycopg` driver, the corresponding Psycopg 3 dependencies must be installed.

### FileNotFoundError in a Notebook

Check the notebook's current working directory:

```python
from pathlib import Path
print(Path.cwd())
```

The notebooks use data paths relative to the `notebooks/` folder.

### Missing PostgreSQL Tables

Execute the EDA notebook's database-loading sections before running its SQL queries.

The database tables must be populated before they can be queried.

### Different Statistics Canada Record Count

The Statistics Canada API provides live data that may be updated with new monthly observations.

For reproducing the documented analysis, use the original downloaded CSV included with the repository.

---

## 17. Workshop Submission

This project is submitted individually as part of the Linear Regression Architecture Workshop.

**GitHub Repository:**

https://github.com/UbaidUllah777/LinearRegressionArchitecture_Workshop

**Git Repository URL:**

```text
https://github.com/UbaidUllah777/LinearRegressionArchitecture_Workshop.git
```

The repository contains the implemented workshop pipeline, required notebooks, modular Python scripts, configuration, experiment results, documentation, and proposed future architectural extensions.
