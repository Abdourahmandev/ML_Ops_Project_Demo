# Projet MLOps - Suivi d'experiences avec MLflow

Ce projet compare trois modeles de regression (ElasticNet, Ridge, Lasso) pour predire la qualite du vin rouge a partir de variables physico-chimiques, avec suivi complet dans MLflow.

Contexte: Examen MLOps / MLflow Tracking (Databricks Free Edition, compute serverless).

## Objectif

- Entrainer 3 familles de modeles sur le dataset wine quality
- Logger chaque run dans MLflow (params, metrics, modele, artefacts, tags)
- Comparer les performances (RMSE, MAE, R2)

Resultat attendu: 3 experiences x 4 configurations = 12 runs MLflow.

## Structure du depot

```
ML_Ops_Project_Demo/
|- train_mlflow.py
|- explication_mlflow.txt
|- Readme.md
|- data/
|  \- readme.md
\- captures_mlflow/
   |- capture_experiments.PNG
   |- experiments_run.jpeg
   |- exp_multi_EL.PNG
   |- exp_multi_Ridge.PNG
   \- exp_multi_Lasso.PNG
```

## Execution sur Databricks

1. Deposer red-wine-quality.csv dans un Volume Unity Catalog.
   Exemple: /Volumes/workspace/default/ml_ops_data/red-wine-quality.csv
2. Ouvrir train_mlflow.py dans Databricks comme notebook Python.
3. Verifier la variable SOURCE_CSV dans le script.
4. Lancer le notebook.
5. Ouvrir AI/ML -> Experiments pour visualiser les runs.

Le script:
- cree les experiences MLflow par utilisateur (/Users/<email>/exp_multi_*)
- effectue un split train/test
- sauvegarde train.csv et test.csv dans le Volume Unity Catalog
- logge ces CSV en artefacts MLflow pour chaque run

## Hyperparametres

- Arguments disponibles: --alpha, --l1_ratio
- Configurations definies dans RUN_CONFIGS (4 runs par modele)

## Fichiers de resultats

- Analyse et interpretation: explication_mlflow.txt
- Captures d'ecran MLflow: dossier captures_mlflow/

## Stack technique

Python, pandas, numpy, scikit-learn, MLflow, Databricks Free Edition.

## Notes importantes

- Pas de set_tracking_uri necessaire dans Databricks serverless.
- parse_known_args est utilise pour compatibilite notebook.
- Le chemin du Volume dans ce projet est ml_ops_data (avec underscore).