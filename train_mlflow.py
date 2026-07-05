# train_mlflow # Databricks notebook source
# =====================================================================
# train_mlflow.py  —  DEPLOIEMENT SUR DATABRICKS FREE EDITION
# 3 experiences (ElasticNet, Ridge, Lasso) x 4 runs = 12 runs MLflow
# =====================================================================
 
import argparse, logging, os, shutil, warnings
import mlflow, mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
 
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
 
# argparse : parse_known_args() pour ne pas planter dans un notebook
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args, _ = parser.parse_known_args()
 
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
 
COMMON_TAGS = {
    "engineering": "ML platform",
    "release.candidate": "RC1",
    "release.version": "2.0",
    "project": "wine-quality-regression",
    "exam": "mlops",
}
 
def make_elasticnet(alpha, l1_ratio):
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42), {"alpha": alpha, "l1_ratio": l1_ratio}
def make_ridge(alpha, l1_ratio):
    return Ridge(alpha=alpha, random_state=42), {"alpha": alpha}
def make_lasso(alpha, l1_ratio):
    return Lasso(alpha=alpha, random_state=42), {"alpha": alpha}
 
def train_one_run(run_name, factory, alpha, l1_ratio,
                  train_x, train_y, test_x, test_y):
    mlflow.start_run(run_name=run_name)
    mlflow.set_tags(COMMON_TAGS)
    cur = mlflow.active_run()
    print(f">>> {cur.info.run_name} | id={cur.info.run_id}")
 
    estimator, params = factory(alpha, l1_ratio)
    estimator.fit(train_x, train_y)
    preds = estimator.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, preds)
 
    print(f"    {type(estimator).__name__} {params} -> RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
 
    mlflow.log_params(params)
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.sklearn.log_model(estimator, name="model")
    mlflow.log_artifacts("data/")
    mlflow.end_run()
 
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
 
    # MLflow est integre : on n'appelle PAS set_tracking_uri
    print("Tracking URI:", mlflow.get_tracking_uri())
 
    os.chdir("/tmp")                 # dossier inscriptible
    os.makedirs("data", exist_ok=True)
 
    # >>> ADAPTER ce chemin au Volume ou vous avez depose le CSV <<<
    SOURCE_CSV = "/Volumes/workspace/default/ml_ops_data/red-wine-quality.csv"
    shutil.copy(SOURCE_CSV, "data/red-wine-quality.csv")
 
    data = pd.read_csv("data/red-wine-quality.csv")
    # Si une seule colonne / KeyError 'quality', decommentez :
    # data = pd.read_csv("data/red-wine-quality.csv", sep=";")
 
    train, test = train_test_split(data)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
 
    train_x = train.drop(["quality"], axis=1); test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]];              test_y = test[["quality"]]
 
    # Nom d'experience = chemin absolu /Users/<courriel>/...
    try:
        email = spark.sql("SELECT current_user()").collect()[0][0]
        EXP_PREFIX = f"/Users/{email}/"
    except Exception:
        EXP_PREFIX = ""
 
    EXPERIMENTS = [
        ("exp_multi_EL", make_elasticnet),
        ("exp_multi_Ridge", make_ridge),
        ("exp_multi_Lasso", make_lasso),
    ]
    RUN_CONFIGS = [
        {"alpha": args.alpha, "l1_ratio": args.l1_ratio},
        {"alpha": 0.9, "l1_ratio": 0.7},
        {"alpha": 0.4, "l1_ratio": 0.5},
        {"alpha": 0.1, "l1_ratio": 0.2},
    ]
 
    for exp_name, factory in EXPERIMENTS:
        print(f"\n=== Experiment: {exp_name} ===")
        exp = mlflow.set_experiment(f"{EXP_PREFIX}{exp_name}")
        print(f"    id={exp.experiment_id}")
        for i, cfg in enumerate(RUN_CONFIGS, start=1):
            train_one_run(f"run{i}.1", factory, cfg["alpha"], cfg["l1_ratio"],
                          train_x, train_y, test_x, test_y)
 
    last = mlflow.last_active_run()
    print("\nDernier run:", last.info.run_name, last.info.run_id)
