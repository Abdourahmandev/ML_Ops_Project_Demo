# Explication MLflow — Examen 3 — MLOps et MLflow Tracking (Partie 2)

> **Nom :** Abdourahman  
> **Plateforme :** Databricks Free Edition (Serverless)

---

## Étape 1 — Rôle des imports

| Bibliothèque | Rôle |
|---|---|
| `argparse` | Lit les arguments passés en ligne de commande (ex. `--alpha 0.5`). Évite de modifier le code à chaque test. |
| `logging` | Affiche des messages de suivi (infos, avertissements, erreurs). Plus propre que d'utiliser `print` partout. |
| `os` | Interagit avec le système de fichiers : créer des dossiers, lire des variables d'environnement, construire des chemins. |
| `mlflow` | Bibliothèque principale du projet. Enregistre les expériences, paramètres, métriques et modèles pour les comparer. |
| `pandas` | Charge et manipule les données sous forme de tableau (DataFrame). Utilisé pour lire le CSV et faire le split. |
| `numpy` | Calcul numérique. Utilisé pour calculer la racine carrée dans la formule RMSE. |
| `scikit-learn` | Contient les algorithmes ML (ElasticNet, Ridge, Lasso) et les métriques d'évaluation. |

---

## Étape 2 — Pourquoi utiliser `logging` ?

On utilise `logging` plutôt que `print` parce que c'est beaucoup plus flexible. Avec `logging`, on peut choisir le niveau de verbosité (`DEBUG`, `INFO`, `WARNING`, `ERROR`) et activer ou désactiver les messages sans toucher au code. Dans ce projet, on utilise le niveau `WARN` pour ne pas surcharger la sortie avec des messages inutiles. C'est une bonne pratique surtout quand on travaille en équipe ou en production, parce que les logs peuvent aussi être redirigés vers des fichiers ou des outils de monitoring.

---

## Étape 3 — Arguments en ligne de commande (`argparse`)

**1) À quoi sert `--alpha` ?**

`alpha` contrôle la force de la régularisation. Plus alpha est grand, plus le modèle est pénalisé et plus les coefficients tendent vers zéro. Un petit alpha donne un modèle plus libre, un grand alpha simplifie davantage le modèle pour éviter le surapprentissage.

**2) À quoi sert `--l1_ratio` ?**

`l1_ratio` est utilisé uniquement dans ElasticNet. Il détermine le mélange entre la régularisation L1 (Lasso) et L2 (Ridge) :
- `l1_ratio = 1` → le modèle se comporte comme Lasso (zéros dans les poids)
- `l1_ratio = 0` → le modèle se comporte comme Ridge (poids petits mais non nuls)
- Entre 0 et 1 → mélange des deux

**3) Pourquoi est-il utile de modifier ces valeurs en ligne de commande dans un contexte MLOps ?**

En MLOps, on veut automatiser les entraînements et les comparaisons. Si les hyperparamètres sont codés en dur dans le script, il faut modifier le fichier à chaque test. Avec `argparse`, on peut lancer le script avec différentes valeurs sans toucher au code, ce qui facilite l'intégration dans des pipelines automatiques (ex. GitHub Actions, Databricks Jobs). Ça rend aussi le code plus propre et plus réutilisable.

---

## Étape 4 — Métriques d'évaluation

| Métrique | Description | Valeur préférable |
|---|---|---|
| **RMSE** (Root Mean Squared Error) | Racine carrée de la moyenne des erreurs au carré. Pénalise fortement les grosses erreurs. | ↓ La plus petite possible (0 = parfait) |
| **MAE** (Mean Absolute Error) | Moyenne des valeurs absolues des erreurs. Plus robuste aux valeurs aberrantes que le RMSE. | ↓ La plus petite possible |
| **R²** (Coefficient de détermination) | Proportion de variance de la cible expliquée par le modèle. R²=1 → parfait, R²=0 → pas mieux que la moyenne. | ↑ Le plus grand possible (idéalement proche de 1) |

---

## Étape 5 — Tags MLflow

Les tags dans MLflow servent à annoter les runs avec des informations contextuelles qui ne sont pas des hyperparamètres ni des métriques. Par exemple, on peut indiquer la version du projet, le nom de l'équipe, l'environnement d'exécution ou la phase du développement.

**Exemple d'utilisation en équipe :**

On a défini les tags `"release.version": "2.0"` et `"release.candidate": "RC1"`. Si plusieurs développeurs tournent des expériences en parallèle, les tags permettent de filtrer rapidement dans MLflow tous les runs d'une version donnée, sans mélanger les résultats de dev avec ceux de staging ou prod. On pourrait aussi ajouter un tag `"auteur": "prenom"` pour savoir qui a lancé quel run.

---

## Étape 6 — Fonctions de création des modèles

**1) Pourquoi créer une fonction par modèle ?**

Ça permet de séparer la logique propre à chaque modèle. Si demain je veux changer les paramètres de Ridge sans toucher à ElasticNet, je modifie juste la fonction `make_ridge`. C'est plus lisible et plus facile à maintenir. Ça suit le principe de responsabilité unique.

**2) Pourquoi ElasticNet utilise `alpha` et `l1_ratio` ?**

ElasticNet combine la pénalité L1 (Lasso) et L2 (Ridge). Il a donc besoin de deux paramètres :
- `alpha` pour contrôler l'intensité globale de la régularisation
- `l1_ratio` pour doser le mélange entre L1 et L2

Sans `l1_ratio`, on ne pourrait pas distinguer ElasticNet de Ridge ou Lasso.

**3) Pourquoi Ridge et Lasso enregistrent surtout `alpha` ?**

Ridge n'utilise que la régularisation L2, donc le seul paramètre à contrôler est `alpha`. Idem pour Lasso qui n'utilise que L1. Il n'y a pas de "ratio" à ajuster puisqu'il n'y a qu'un seul type de pénalité. Enregistrer `l1_ratio` pour ces modèles n'aurait pas de sens.

---

## Étape 7 — Rôle de chaque appel MLflow dans `train_one_run`

| Appel MLflow | Rôle |
|---|---|
| `mlflow.start_run(run_name=...)` | Démarre un nouveau run avec un nom lisible. Tout ce qui est loggé après est associé à ce run. |
| `mlflow.set_tags(COMMON_TAGS)` | Applique les métadonnées communes (version, projet, équipe). Apparaissent dans l'interface MLflow pour filtrer. |
| `mlflow.log_params(params)` | Enregistre les hyperparamètres (ex. `alpha`, `l1_ratio`). On sait exactement avec quoi le modèle a été entraîné. |
| `mlflow.log_metrics({...})` | Enregistre les métriques de performance (RMSE, MAE, R²) calculées sur le jeu de test. Sert à comparer les runs. |
| `mlflow.sklearn.log_model(estimator, "model")` | Sérialise et sauvegarde le modèle dans MLflow. On peut le recharger directement pour faire des prédictions. |
| `mlflow.log_artifacts("data/")` | Sauvegarde des fichiers (CSV train/test) comme artefacts du run. Garantit la traçabilité des données. |
| `mlflow.end_run()` | Ferme le run proprement. Sans cet appel, le run resterait "actif" et les prochains logs pourraient s'y ajouter par erreur. |

---

## Étape 9 — Pourquoi plusieurs runs avec différentes valeurs ?

On lance plusieurs runs avec différentes valeurs d'`alpha` et de `l1_ratio` pour trouver la combinaison d'hyperparamètres qui donne les meilleures performances. Sans cette comparaison, on choisit les paramètres au hasard et on ne sait pas si le modèle aurait pu faire mieux.

En MLOps, cette pratique s'appelle le **hyperparameter tuning**. MLflow est très utile dans ce cas parce qu'il enregistre tout automatiquement et permet de comparer visuellement tous les runs dans une même expérience. C'est beaucoup plus rapide que de noter les résultats à la main dans un tableau Excel.

---

## Étape 12 — Tableau des résultats

> Les valeurs ci-dessous proviennent des captures MLflow.  
> Le run1.1 de Ridge (alpha=0.7) est confirmé par la capture `experiments_run.jpeg`.  
> Les autres valeurs ont été lues dans l'interface MLflow lors de l'exécution.

| Expérience | Run | Modèle | Alpha | L1 ratio | RMSE | MAE | R² |
|---|---|---|---|---|---|---|---|
| exp_multi_EL | run1.1 | ElasticNet | 0.7 | 0.7 | 0.6740 | 0.5244 | 0.2735 |
| exp_multi_EL | run4.1 | ElasticNet | 0.1 | 0.2 | 0.6095 | 0.4820 | 0.3980 |
| exp_multi_Ridge | run1.1 | Ridge | 0.7 | N/A | 0.6247 | 0.5013 | 0.3691 |
| exp_multi_Ridge | run4.1 | Ridge | 0.1 | N/A | 0.5990 | 0.4715 | **0.4120** |
| exp_multi_Lasso | run1.1 | Lasso | 0.7 | N/A | 0.7183 | 0.5618 | 0.1672 |
| exp_multi_Lasso | run4.1 | Lasso | 0.1 | N/A | 0.6398 | 0.5021 | 0.3540 |

---

## Étape 13 — Questions d'analyse des résultats

**1) Quel modèle donne le plus petit RMSE ?**

Ridge avec `alpha=0.1` (run4.1) donne le plus petit RMSE (~0.5990). En général, Ridge performe mieux que Lasso sur ce dataset parce qu'il garde tous les coefficients au lieu d'en forcer certains à zéro. Avec un alpha faible, la régularisation est plus douce et le modèle colle mieux aux données.

**2) Quel modèle donne le meilleur R² ?**

Ridge (run4.1, alpha=0.1) donne aussi le meilleur R² (~0.41). Ça confirme que Ridge avec peu de régularisation est le plus adapté à ce problème. L'ElasticNet avec alpha faible s'en approche aussi.

**3) Est-ce que tous les modèles donnent exactement les mêmes résultats ?**

Non, les résultats sont différents selon le modèle et les hyperparamètres. Lasso donne les pires résultats avec un alpha élevé car il force beaucoup de coefficients à zéro, ce qui perd de l'information sur des features qui sont peut-être toutes utiles dans ce dataset. Ridge garde tous les coefficients petits mais non nuls, ce qui lui permet de mieux capturer les relations entre les features et la qualité du vin.

**4) Quel paramètre semble influencer les résultats ?**

`alpha` a clairement le plus gros impact. Quand alpha est grand (0.7 ou 0.9), les performances se dégradent pour tous les modèles. Quand alpha est petit (0.1), tous les modèles s'améliorent. `l1_ratio` influence surtout ElasticNet, mais son effet est moins prononcé que celui d'alpha sur les résultats finaux.

**5) Pourquoi MLflow aide à comparer les modèles ?**

Sans MLflow, je devrais noter les résultats à la main dans un tableau, et je risquerais d'oublier avec quels paramètres chaque modèle a été entraîné. MLflow enregistre tout automatiquement : les paramètres, les métriques, les artefacts et les modèles. Dans l'interface, on peut filtrer, trier et visualiser les runs en un coup d'œil. Ça fait gagner beaucoup de temps et rend les comparaisons fiables et reproductibles.

**6) Quel modèle choisiriez-vous pour continuer le projet ? Pourquoi ?**

Je choisirais **Ridge avec alpha=0.1** parce qu'il donne le meilleur RMSE et le meilleur R² parmi tous les runs. C'est aussi un modèle simple et interprétable, ce qui est un avantage si on veut expliquer les résultats. Lasso ne convient pas bien ici car il élimine trop d'informations avec une forte régularisation. ElasticNet pourrait être intéressant si on faisait une recherche plus fine sur alpha et l1_ratio.

---

## Étape 14 — Questions de compréhension

**1) À quoi sert `if __name__ == "__main__"` ?**

Cette condition permet de s'assurer que le bloc de code principal ne s'exécute que si le fichier est lancé directement (ex. `python train_mlflow.py`). Si le fichier est importé comme module dans un autre script, ce bloc ne s'exécute pas. C'est une bonne pratique en Python pour rendre les fichiers à la fois utilisables comme script ET comme module importable.

**2) À quoi sert `warnings.filterwarnings("ignore")` ?**

scikit-learn et d'autres bibliothèques peuvent afficher des avertissements pendant l'entraînement (par exemple si Lasso ne converge pas parfaitement avec certains paramètres). Ces messages peuvent encombrer la sortie. `filterwarnings("ignore")` les masque pour que la sortie reste lisible. En production, on voudrait plutôt les capturer avec `logging` que les ignorer.

**3) À quoi sert `np.random.seed(40)` ?**

Cela fixe la graine du générateur aléatoire de numpy. Sans cette ligne, chaque exécution du script pourrait donner des résultats légèrement différents à cause de l'aléatoire dans certains algorithmes. Avec une graine fixe, les résultats sont reproductibles : en relançant exactement le même script, on obtient exactement les mêmes métriques.

**4) Quelle est la différence entre un paramètre et une métrique ?**

Un **paramètre** est une valeur qu'on choisit **avant** l'entraînement et qui configure le modèle (ex. `alpha=0.7`). Une **métrique** est une valeur qu'on calcule **après** l'entraînement pour mesurer la performance du modèle (ex. `RMSE=0.6247`). Les paramètres sont des entrées, les métriques sont des sorties.

**5) Quelle est la différence entre `mlflow.log_params()` et `mlflow.log_metrics()` ?**

`log_params()` enregistre les hyperparamètres (valeurs fixes choisies avant l'entraînement). `log_metrics()` enregistre les performances calculées après l'entraînement. Dans l'interface MLflow, les deux apparaissent dans des sections séparées. On peut filtrer et trier les runs selon les métriques (ex. trier par RMSE croissant).

**6) Quelle est la différence entre `mlflow.sklearn.log_model()` et `mlflow.log_artifacts()` ?**

`log_model()` est spécifique aux modèles scikit-learn : il sauvegarde le modèle avec sa signature MLflow, ce qui permet de le recharger directement pour faire des prédictions via `mlflow.pyfunc.load_model()`. `log_artifacts()` sauvegarde n'importe quel fichier (CSV, images, logs) comme pièce jointe du run. Ce n'est pas un modèle déployable, juste un fichier de référence.

**7) Pourquoi faut-il séparer les données en train et test ?**

On sépare les données pour évaluer honnêtement la capacité du modèle à généraliser sur des données qu'il n'a jamais vues. Si on évaluait le modèle sur les mêmes données qu'il a apprises, il aurait un score artificiellement bon (surapprentissage). Le jeu de test simule des données réelles en production.

**8) Pourquoi la colonne `quality` ne doit pas être présente dans `train_x` ?**

`quality` est la colonne cible qu'on cherche à prédire. Si on l'incluait dans les features, le modèle "tricherait" en apprenant directement la réponse. Ça donnerait un R² de 1 en entraînement mais le modèle serait complètement inutilisable en production où on ne connaît pas la qualité à l'avance.

**9) Pourquoi sauvegarde-t-on `train.csv` et `test.csv` ?**

Pour garantir la reproductibilité des expériences. Si on relance le script plus tard avec un nouveau split, les résultats changeraient. En sauvegardant les splits utilisés et en les loggant comme artefacts MLflow, on peut toujours retrouver exactement quelles données ont servi à entraîner chaque modèle.

**10) Pourquoi MLflow est utile dans un projet MLOps réel ?**

En MLOps, les équipes font tourner des dizaines ou des centaines d'expériences. Sans outil de tracking, on perd rapidement la trace de ce qui a été testé. MLflow centralise tout : les paramètres, les métriques, les modèles et les artefacts. Ça facilite la collaboration en équipe, la comparaison des modèles et la promotion d'un modèle vers la production via le Model Registry.

**11) Pourquoi un bon score sur le jeu de test ne suffit pas toujours pour déployer un modèle en production ?**

Un bon RMSE sur le test ne garantit pas que le modèle fonctionnera bien en production. Les données réelles peuvent avoir une distribution différente du jeu de test (concept drift). Le modèle pourrait aussi être lent, trop lourd, ou non-robuste aux valeurs manquantes. Il faut aussi valider que le modèle respecte des contraintes métier (ex. ne pas prédire une qualité négative) et tester son comportement sous charge avant de le déployer.

**12) Que faudrait-il ajouter pour rendre ce projet plus professionnel ?**

- **Docker** : pour packager l'environnement d'exécution et garantir que le script tourne de la même façon partout (dev, staging, prod).
- **GitHub Actions** : pour automatiser les tests et le réentraînement du modèle à chaque push sur la branche principale.

Ces deux éléments permettraient d'avoir un vrai pipeline CI/CD pour le ML, ce qui est la base d'un projet MLOps mature.
