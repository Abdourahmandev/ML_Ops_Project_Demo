# Projet MLOps — Suivi d'expériences avec MLflow

Comparaison de trois modèles de régression (**ElasticNet**, **Ridge**, **Lasso**) pour
prédire la qualité du vin rouge à partir de variables physico-chimiques, avec suivi complet
des expériences dans **MLflow**. Réalisé sur **Databricks Free Edition** (compute serverless).

> Examen 3 — MLOps et MLflow Tracking, partie 2 (laboratoire évalué).

## Aperçu

- **Dataset** : `red-wine-quality.csv` — cible : `quality`
- **Modèles** : ElasticNet, Ridge, Lasso (scikit-learn)
- **Suivi** : MLflow (paramètres, métriques, modèle, artefacts, tags)
- **Résultat** : 3 expériences × 4 runs = **12 runs MLflow**

## Structure du dépôt

```
mlops-exam-mlflow/
├── train_mlflow.py          # Script d'entraînement + logging MLflow (notebook Databricks)
├── explication_mlflow.txt   # Réponses aux questions et analyse des résultats
├── README.md
└── .gitignore
```

Les données (`data/`), les runs et les modèles MLflow ne sont **pas** versionnés :
la source vit dans un Volume Unity Catalog et les artefacts sont gérés par Databricks.

## Exécution (Databricks Free Edition)

1. Déposer `red-wine-quality.csv` dans un Volume Unity Catalog
   (ex. `/Volumes/workspace/default/mlops_data/`).
2. Ouvrir `train_mlflow.py` comme **notebook**, compute = **Serverless**.
3. Adapter la variable `SOURCE_CSV` au chemin exact du Volume.
4. Lancer la cellule → **3 expériences × 4 runs = 12 runs**.
5. Consulter les runs dans le menu latéral **AI/ML → Experiments**.

### Hyperparamètres

Réglables via `--alpha` et `--l1_ratio` (valeurs `default` dans `argparse`), ou en modifiant
la liste `RUN_CONFIGS` dans le script.

## Résultats

Le tableau comparatif (RMSE, MAE, R² par modèle et par run) ainsi que l'analyse
figurent dans `explication_mlflow.txt`.

## Pile technique

Python · scikit-learn · pandas · numpy · MLflow · Databricks Free Edition

## Notes d'adaptation à Databricks

Le script diffère légèrement d'un MLflow local :

- MLflow est **intégré** — pas de `set_tracking_uri` (l'URI vaut `databricks`).
- Noms d'expériences en **chemin absolu** : `/Users/<courriel>/exp_multi_*`.
- Arguments lus via `parse_known_args()` (compatible notebook).
- Dossier de travail `/tmp` inscriptible pour les fichiers générés.