# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Entraînement MLflow Multi-Modèles
# MAGIC
# MAGIC Ce notebook entraîne **3 types de modèles** de régression (ElasticNet, Ridge, Lasso) sur le dataset de qualité du vin rouge.
# MAGIC
# MAGIC **Objectif** : Générer 12 runs MLflow (3 expériences × 4 configurations) pour comparer les performances.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Imports
import argparse, logging, os, shutil, warnings
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
import mlflow, mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
 
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# COMMAND ----------

# DBTITLE 1,Configuration & Fonctions utilitaires
# MAGIC %md
# MAGIC ## Configuration des paramètres, définitions de fonctions
# MAGIC
# MAGIC Cette section configure les paramètres génériques du notebook et définit les fonctions utilitaires pour l'évaluation des métriques et la préparation des modèles.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Paramètres, tags et fonctions utilitaires
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
    
    # ✅ NOUVEAU : Logger les CSV depuis le Volume UC comme artefacts MLflow
    # Créer un dossier temporaire pour les artefacts
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Copier les CSV depuis le Volume UC vers le dossier temporaire
        volume_path = "/Volumes/workspace/default/ml_ops_data/"
        shutil.copy(f"{volume_path}train.csv", os.path.join(data_dir, "train.csv"))
        shutil.copy(f"{volume_path}test.csv", os.path.join(data_dir, "test.csv"))
        
        # Logger les artefacts dans MLflow
        mlflow.log_artifacts(data_dir, artifact_path="data")
        print(f"    ✅ Artefacts CSV loggés dans MLflow (data/train.csv, data/test.csv)")
    
    mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,Préparation des données
# MAGIC %md
# MAGIC ## Chargement & préparation du dataset
# MAGIC
# MAGIC Cette section charge le CSV, prépare les dossiers, fait le split train/test et formate les données pour l'entraînement.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Chargement, split et préparation des features/target
warnings.filterwarnings("ignore")
np.random.seed(40)

print("Tracking URI:", mlflow.get_tracking_uri())

# ✅ NOUVEAU : Utiliser le Volume UC comme destination PERSISTANTE
VOLUME_PATH = "/Volumes/workspace/default/ml_ops_data/"
SOURCE_CSV = f"{VOLUME_PATH}red-wine-quality.csv"

print(f"📂 Chargement depuis Volume UC: {SOURCE_CSV}")
data = pd.read_csv(SOURCE_CSV)

# Split train/test avec random_state fixe pour reproductibilité
train, test = train_test_split(data, random_state=42)

# ✅ NOUVEAU : Sauvegarder dans le Volume UC (PERSISTANT)
train_path = f"{VOLUME_PATH}train.csv"
test_path = f"{VOLUME_PATH}test.csv"

print(f"💾 Sauvegarde de train.csv dans le Volume UC...")
train.to_csv(train_path, index=False)
print(f"   ✅ {train_path} ({len(train)} lignes)")

print(f"💾 Sauvegarde de test.csv dans le Volume UC...")
test.to_csv(test_path, index=False)
print(f"   ✅ {test_path} ({len(test)} lignes)")

print(f"\n🎯 Fichiers train/test maintenant PERSISTANTS dans Unity Catalog!\n")

# Préparation des features/target pour l'entraînement
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# COMMAND ----------

# DBTITLE 1,Configuration des expériences MLflow
# MAGIC %md
# MAGIC ## Configuration des expériences et runs MLflow
# MAGIC
# MAGIC Définition des expériences et des configurations d'entraînement pour chaque modèle.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Définition des expériences et configs de runs
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
EXP_PREFIX = f"/Users/{username}/"

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

# COMMAND ----------

# DBTITLE 1,Boucle d'entraînement et suivi MLflow
# MAGIC %md
# MAGIC ## Entraînement des modèles & tracking MLflow
# MAGIC
# MAGIC Exécution de la boucle d'entraînement : chaque expérience MLflow est lancée avec différents hyperparamètres, métriques loggées & artefacts sauvegardés.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Boucle d'entraînement multi-expériences MLflow
# mlflow.set_tracking_uri("databricks")  # Pas nécessaire en Serverless - utilisé par défaut

for exp_name, factory in EXPERIMENTS:
    print(f"\n=== Experiment: {exp_name} ===")
    exp = mlflow.set_experiment(f"{EXP_PREFIX}{exp_name}")
    print(f"    id={exp.experiment_id}")
    for i, cfg in enumerate(RUN_CONFIGS, start=1):
        train_one_run(f"run{i}.1", factory, cfg["alpha"], cfg["l1_ratio"],
                      train_x, train_y, test_x, test_y)

last = mlflow.last_active_run()
print("\nDernier run:", last.info.run_name, last.info.run_id)

# COMMAND ----------

# DBTITLE 1,Vérification finale : Fichiers persistants
# 🔍 VÉRIFICATION FINALE : Fichiers train.csv et test.csv
import os

print("\n" + "="*80)
print("🔍 VÉRIFICATION DES FICHIERS PERSISTANTS")
print("="*80)

VOLUME_PATH = "/Volumes/workspace/default/ml_ops_data/"

# Lister tous les fichiers du Volume
print(f"\n📁 Contenu du Volume UC: {VOLUME_PATH}")
print("-"*80)
files = dbutils.fs.ls(VOLUME_PATH)
for f in files:
    size_kb = f.size / 1024
    print(f"   {'  📄' if not f.isDir() else '📂'} {f.name:<30} {size_kb:>10.2f} KB")

print("\n" + "="*80)
print("✅ RÉSULTAT")
print("="*80)

# Vérifier l'existence des fichiers
train_exists = any(f.name == "train.csv" for f in files)
test_exists = any(f.name == "test.csv" for f in files)

if train_exists and test_exists:
    print("\n✅ SUCCÈS ! Les fichiers train.csv et test.csv sont PERSISTANTS dans Unity Catalog")
    print(f"\n   📍 Chemin : {VOLUME_PATH}")
    print(f"   🔒 Accessibles depuis n'importe quel notebook/compute")
    print(f"   📈 Traçabilité : Aussi loggés comme artefacts MLflow dans chaque run")
else:
    print("\n⚠️  Fichiers manquants :")
    if not train_exists:
        print("   ❌ train.csv")
    if not test_exists:
        print("   ❌ test.csv")

print("\n" + "="*80)
