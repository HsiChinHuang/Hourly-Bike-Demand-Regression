# 🚲 Hourly Bike Demand Regression

## Problem Description
Bike-sharing systems are highly dynamic. Accurately predicting hourly bike demand is crucial for fleet management and redistribution. This project aims to predict the hourly rental count (`cnt`) using weather and temporal features. We treat this as a regression problem and evaluate models using RMSE.

## Data
The dataset is the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (2011–2012). The `hour.csv` file (17,379 rows) is used. 
*Note: We explicitly drop `casual` and `registered` because they are decompositions of the target `cnt` and would cause data leakage.*

## Data Preparation & EDA (Key Findings)
- **Commuting vs. Leisure**: Working days show two sharp peaks (8 AM and 5-6 PM), while weekends/holidays show a single broad peak in the afternoon.
- **Weather Sensitivity**: Clear weather yields a median of 159 rentals/hour, while light rain/snow cuts it down to 63 (~60% drop).
- **Collinearity**: `temp` and `atemp` have a correlation of 0.99, so `atemp` was dropped.

## Feature Engineering Experiment (The Core Highlight 🌟)
Instead of using the `hr` feature as raw numbers, we tested three encoding methods using XGBoost:
1. **v1 (Raw Numerical)**: RMSE = 42.94
2. **v2 (One-Hot)**: RMSE = 54.37 (Tree models struggle with sparse high-dimensional categories)
3. **v3 (Sin/Cos Cyclical)**: **RMSE = 42.39 (Winner 🏆)**

By mapping hours to a 24-hour cycle using `sin` and `cos`, the model learns that 23:00 and 00:00 are adjacent.

## Model Selection
We trained Random Forest and XGBoost. XGBoost with the **v3 Sin/Cos encoding** achieved the best **Test RMSE of 39.81**.

## Project Structure
- `notebook.ipynb`: EDA, feature engineering experiments, and model tuning.
- `train.py`: Trains the final XGBoost model and saves `model.json` + `dv.pkl`.
- `predict.py`: Flask web application for serving predictions.
- `Dockerfile`: For containerizing the web service.

## How to Run

### Option A: Using `uv` (Recommended)
1. `uv sync`
2. `uv run python train.py` (to generate the model)
3. `uv run python predict.py` (to start the web server)

### Option B: Using standard `pip` (For Notebook & Training)
1. Create venv: `python -m venv venv`
2. Activate: `source venv/bin/activate` (Windows: `venv\Scripts\activate`)
3. Install all dependencies: `pip install -r requirements.txt`
4. Run Notebook: `jupyter notebook`
5. Train model: `python train.py`

### Option C: Using Docker (For Deployment only)
*Note: Docker uses `requirements-prod.txt` to keep the image lightweight.*
1. Build: `docker build -t bike-demand .`
2. Run: `docker run -p 5000:5000 bike-demand`

## Deployment
The service is deployed at: [Insert Render URL Here]