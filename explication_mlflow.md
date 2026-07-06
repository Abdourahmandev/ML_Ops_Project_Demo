# Explication MLflow — Examen 3 — MLOps et MLflow Tracking

> **Nom :** Abdourahman
> **Plateforme :** Databricks Free Edition / Serverless

---

## Étape 1 — Rôle des imports

Dans ce projet, on utilise plusieurs bibliothèques Python. Chacune a un rôle précis dans le notebook.

| Bibliothèque     | Rôle                                                                                                                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `argparse`       | Sert à passer des paramètres au script, comme `--alpha` ou `--l1_ratio`, sans modifier le code à chaque fois.                                                                                                 |
| `logging`        | Sert à gérer les messages affichés pendant l’exécution. C’est plus propre que de seulement utiliser `print`, même si dans mon notebook j’utilise encore plusieurs `print` pour voir les résultats rapidement. |
| `os`             | Sert à travailler avec les chemins, les dossiers et les variables d’environnement.                                                                                                                            |
| `shutil`         | Sert à copier les fichiers, par exemple les fichiers train/test vers un dossier temporaire avant de les logger dans MLflow.                                                                                   |
| `warnings`       | Sert à cacher certains avertissements pour rendre l’affichage plus lisible.                                                                                                                                   |
| `mlflow`         | Sert à suivre les expériences, les paramètres, les métriques, les modèles et les artefacts.                                                                                                                   |
| `mlflow.sklearn` | Sert à sauvegarder un modèle scikit-learn dans MLflow.                                                                                                                                                        |
| `pandas`         | Sert à lire et manipuler les données sous forme de DataFrame.                                                                                                                                                 |
| `numpy`          | Sert ici surtout pour le calcul du RMSE avec la racine carrée.                                                                                                                                                |
| `scikit-learn`   | Sert pour les modèles de régression, le split train/test et les métriques.                                                                                                                                    |

---

## Étape 2 — Pourquoi utiliser `logging` ?

On utilise `logging` parce que c’est plus flexible que `print`. Avec `logging`, on peut choisir le niveau de message qu’on veut afficher, par exemple `DEBUG`, `INFO`, `WARNING` ou `ERROR`.

Dans mon projet, le niveau est mis à `WARN`, donc normalement on affiche seulement les messages importants. Par contre, comme c’est un notebook d’exercice, j’utilise aussi des `print` pour voir les résultats plus facilement pendant l’exécution.

Dans un vrai projet MLOps, `logging` serait plus utile parce qu’on pourrait garder les logs dans un fichier ou les envoyer vers un outil de monitoring.

---

## Étape 3 — Arguments en ligne de commande avec `argparse`

### 1) À quoi sert `--alpha` ?

`alpha` sert à contrôler la force de la régularisation.

Si `alpha` est grand, le modèle est plus pénalisé. Les coefficients deviennent plus petits et le modèle devient plus simple. Ça peut aider à éviter le surapprentissage.

Si `alpha` est petit, le modèle est plus libre. Il peut mieux apprendre les données, mais il peut aussi plus facilement surapprendre.

### 2) À quoi sert `--l1_ratio` ?

`l1_ratio` est utilisé surtout par ElasticNet. Il sert à choisir le mélange entre Lasso et Ridge.

* Si `l1_ratio = 1`, on est proche de Lasso.
* Si `l1_ratio = 0`, on est proche de Ridge.
* Si la valeur est entre 0 et 1, c’est un mélange des deux.

Dans mon code, Ridge et Lasso reçoivent aussi `l1_ratio` dans la fonction, mais ils ne l’utilisent pas vraiment. C’est surtout pour garder la même structure dans les fonctions.

### 3) Pourquoi modifier ces valeurs en ligne de commande dans un contexte MLOps ?

En MLOps, on veut souvent tester plusieurs configurations sans changer le code à chaque fois. Avec `argparse`, on peut lancer le même script avec différentes valeurs de `alpha` et `l1_ratio`.

C’est utile si on veut automatiser les entraînements avec un pipeline, par exemple avec Databricks Jobs ou GitHub Actions. Ça rend aussi le script plus réutilisable.

---

## Étape 4 — Métriques d’évaluation

| Métrique | Explication                                                                                        | Objectif                  |
| -------- | -------------------------------------------------------------------------------------------------- | ------------------------- |
| RMSE     | C’est la racine carrée de la moyenne des erreurs au carré. Elle pénalise plus les grosses erreurs. | Plus petit possible       |
| MAE      | C’est la moyenne des erreurs absolues. Elle est simple à comprendre.                               | Plus petit possible       |
| R²       | Indique à quel point le modèle explique la variable cible.                                         | Plus proche de 1 possible |

Un R² de 1 veut dire que le modèle explique parfaitement les données. Un R² de 0 veut dire que le modèle ne fait pas mieux qu’une prédiction simple basée sur la moyenne. Le R² peut même être négatif si le modèle est mauvais.

---

## Étape 5 — Tags MLflow

Les tags dans MLflow servent à ajouter des informations sur un run. Ce ne sont pas des paramètres du modèle ni des métriques.

Dans mon notebook, j’ai utilisé des tags comme :

* `release.version`
* `release.candidate`
* `project`
* `exam`
* `engineering`

Ces tags permettent de mieux retrouver les runs dans MLflow. Par exemple, si on a plusieurs versions du projet, on peut filtrer les runs par version ou par projet.

Dans un travail d’équipe, c’est utile pour ne pas mélanger les runs de développement, de test ou de production.

---

## Étape 6 — Fonctions de création des modèles

### 1) Pourquoi créer une fonction par modèle ?

J’ai créé une fonction pour chaque modèle pour rendre le code plus clair.

Par exemple :

* `make_elasticnet`
* `make_ridge`
* `make_lasso`

Chaque fonction crée son modèle et retourne aussi les paramètres qu’on veut enregistrer dans MLflow.

Ça rend le code plus facile à lire. Si je veux modifier Ridge, je peux aller directement dans la fonction Ridge sans toucher aux autres modèles.

### 2) Pourquoi ElasticNet utilise `alpha` et `l1_ratio` ?

ElasticNet combine deux types de régularisation :

* L1, comme Lasso ;
* L2, comme Ridge.

Donc il a besoin de deux paramètres.

`alpha` contrôle la force générale de la régularisation.
`l1_ratio` contrôle le mélange entre L1 et L2.

### 3) Pourquoi Ridge et Lasso enregistrent surtout `alpha` ?

Ridge utilise seulement la régularisation L2. Lasso utilise seulement la régularisation L1.

Donc, pour ces deux modèles, le paramètre important est surtout `alpha`. Ils n’ont pas besoin de `l1_ratio`.

C’est pour ça que dans MLflow, on enregistre seulement les paramètres vraiment utilisés par le modèle.

---

## Étape 7 — Rôle de chaque appel MLflow dans `train_one_run`

| Appel MLflow                     | Rôle                                                                  |
| -------------------------------- | --------------------------------------------------------------------- |
| `mlflow.start_run(run_name=...)` | Démarre un nouveau run MLflow.                                        |
| `mlflow.set_tags(COMMON_TAGS)`   | Ajoute les tags communs au run.                                       |
| `mlflow.log_params(params)`      | Enregistre les paramètres du modèle.                                  |
| `mlflow.log_metrics(...)`        | Enregistre les métriques comme RMSE, MAE et R².                       |
| `mlflow.sklearn.log_model(...)`  | Sauvegarde le modèle entraîné dans MLflow.                            |
| `mlflow.log_artifacts(...)`      | Sauvegarde des fichiers liés au run, comme `train.csv` et `test.csv`. |
| `mlflow.end_run()`               | Termine le run proprement.                                            |

Dans mon cas, les fichiers `train.csv` et `test.csv` sont sauvegardés dans le Volume UC. Ensuite, ils sont copiés dans un dossier temporaire et ajoutés comme artefacts dans MLflow.

Ça permet de garder une trace des données utilisées pour chaque run.

---

## Étape 9 — Pourquoi lancer plusieurs runs avec différentes valeurs ?

On lance plusieurs runs pour comparer les modèles avec différents paramètres.

Dans mon projet, je compare trois modèles :

* ElasticNet ;
* Ridge ;
* Lasso.

Je les teste aussi avec différentes valeurs de `alpha` et `l1_ratio`.

L’objectif est de voir quel modèle donne les meilleurs résultats. Sans MLflow, il faudrait noter les résultats manuellement, ce qui devient vite difficile.

Avec MLflow, chaque run garde les paramètres, les métriques, les artefacts et le modèle. C’est donc plus simple de comparer.

---

## Étape 12 — Tableau des résultats

Les résultats suivants viennent des runs affichés dans MLflow.

| Expérience      | Run    | Modèle     | Alpha | L1 ratio | RMSE   | MAE    | R²     |
| --------------- | ------ | ---------- | ----- | -------- | ------ | ------ | ------ |
| exp_multi_EL    | run1.1 | ElasticNet | 0.7   | 0.7      | 0.6740 | 0.5244 | 0.2735 |
| exp_multi_EL    | run4.1 | ElasticNet | 0.1   | 0.2      | 0.6095 | 0.4820 | 0.3980 |
| exp_multi_Ridge | run1.1 | Ridge      | 0.7   | N/A      | 0.6247 | 0.5013 | 0.3691 |
| exp_multi_Ridge | run4.1 | Ridge      | 0.1   | N/A      | 0.5990 | 0.4715 | 0.4120 |
| exp_multi_Lasso | run1.1 | Lasso      | 0.7   | N/A      | 0.7183 | 0.5618 | 0.1672 |
| exp_multi_Lasso | run4.1 | Lasso      | 0.1   | N/A      | 0.6398 | 0.5021 | 0.3540 |

---

## Étape 13 — Questions d’analyse des résultats

### 1) Quel modèle donne le plus petit RMSE ?

Le modèle qui donne le plus petit RMSE est Ridge avec `alpha=0.1`.

Son RMSE est environ `0.5990`, ce qui est le meilleur résultat dans le tableau.

### 2) Quel modèle donne le meilleur R² ?

Le meilleur R² est aussi obtenu par Ridge avec `alpha=0.1`.

Le R² est environ `0.4120`. Ce n’est pas un score parfait, mais c’est le meilleur parmi les runs que j’ai comparés.

### 3) Est-ce que tous les modèles donnent exactement les mêmes résultats ?

Non, les modèles ne donnent pas les mêmes résultats.

Les résultats changent selon le modèle et selon la valeur de `alpha`. Par exemple, Lasso avec un alpha élevé donne de moins bons résultats.

Je pense que c’est parce que Lasso peut mettre certains coefficients à zéro. Si plusieurs variables sont utiles pour prédire la qualité du vin, Lasso peut perdre de l’information.

Ridge fonctionne mieux ici parce qu’il garde toutes les variables, mais réduit quand même l’impact des coefficients.

### 4) Quel paramètre semble influencer les résultats ?

Le paramètre qui semble influencer le plus les résultats est `alpha`.

Quand `alpha` est plus petit, les résultats sont meilleurs dans mes runs. Par exemple, les modèles avec `alpha=0.1` performent mieux que ceux avec `alpha=0.7`.

Pour ElasticNet, `l1_ratio` a aussi un effet, mais dans mes résultats, l’impact de `alpha` semble plus important.

### 5) Pourquoi MLflow aide à comparer les modèles ?

MLflow aide parce qu’il garde toutes les informations des expériences au même endroit.

On peut voir :

* les paramètres utilisés ;
* les métriques obtenues ;
* les modèles entraînés ;
* les fichiers sauvegardés comme artefacts.

Ça évite de devoir tout noter à la main. On peut aussi trier les runs par RMSE ou par R² pour trouver rapidement le meilleur modèle.

### 6) Quel modèle choisiriez-vous pour continuer le projet ?

Je choisirais Ridge avec `alpha=0.1`.

C’est le modèle qui donne le meilleur RMSE et le meilleur R² dans mes résultats. Il est aussi simple à comprendre et à expliquer.

Par contre, je ne dirais pas que c’est le modèle final directement. Avant de le déployer, il faudrait tester plus de valeurs de `alpha`, faire une validation croisée et vérifier si le modèle reste stable avec d’autres données.

---

## Étape 14 — Questions de compréhension

### 1) À quoi sert `if __name__ == "__main__"` ?

Cette condition sert à exécuter une partie du code seulement quand le fichier est lancé directement.

Par exemple, si on lance le script avec :

```python
python train_mlflow.py
```

le bloc principal va s’exécuter.

Mais si le fichier est importé dans un autre fichier Python, ce bloc ne s’exécute pas automatiquement. C’est utile pour rendre le code plus réutilisable.

### 2) À quoi sert `warnings.filterwarnings("ignore")` ?

Cette ligne sert à cacher les avertissements.

Dans un notebook d’exercice, ça permet de garder l’affichage plus propre. Par contre, dans un vrai projet, il faut faire attention, parce que certains warnings peuvent être importants.

Par exemple, un warning peut indiquer qu’un modèle n’a pas bien convergé.

### 3) À quoi sert `np.random.seed(40)` ?

`np.random.seed(40)` sert à fixer l’aléatoire de NumPy. Cela aide à avoir des résultats reproductibles.

Dans mon code, la reproductibilité vient aussi du `random_state=42`, qui est utilisé dans le split train/test et dans les modèles.

Donc `np.random.seed(40)` est utile, mais dans ce notebook, le `random_state` joue aussi un rôle très important.

### 4) Quelle est la différence entre un paramètre et une métrique ?

Un paramètre est une valeur qu’on choisit avant l’entraînement.

Exemple : `alpha=0.7`.

Une métrique est une valeur calculée après l’entraînement pour évaluer le modèle.

Exemple : `RMSE=0.6247`.

Donc les paramètres sont des entrées, et les métriques sont des résultats.

### 5) Quelle est la différence entre `mlflow.log_params()` et `mlflow.log_metrics()` ?

`mlflow.log_params()` sert à enregistrer les paramètres du modèle, comme `alpha`.

`mlflow.log_metrics()` sert à enregistrer les résultats du modèle, comme RMSE, MAE et R².

Dans MLflow, les deux sont séparés, ce qui rend les runs plus faciles à comparer.

### 6) Quelle est la différence entre `mlflow.sklearn.log_model()` et `mlflow.log_artifacts()` ?

`mlflow.sklearn.log_model()` sert à sauvegarder le modèle entraîné.

`mlflow.log_artifacts()` sert à sauvegarder des fichiers liés au run, comme des CSV ou d’autres fichiers de sortie.

Dans mon projet, le modèle est sauvegardé avec `log_model`, alors que les fichiers `train.csv` et `test.csv` sont sauvegardés avec `log_artifacts`.

### 7) Pourquoi faut-il séparer les données en train et test ?

Il faut séparer les données pour évaluer le modèle sur des données qu’il n’a pas vues pendant l’entraînement.

Si on teste le modèle sur les mêmes données que celles utilisées pour l’entraînement, le résultat peut être trop optimiste.

Le jeu de test permet donc de mieux vérifier si le modèle généralise.

### 8) Pourquoi la colonne `quality` ne doit pas être présente dans `train_x` ?

La colonne `quality` est la variable qu’on veut prédire. Elle doit donc être séparée des variables explicatives.

Si on laisse `quality` dans `train_x`, le modèle aurait directement accès à la réponse. Ce serait une fuite de données.

Les résultats seraient donc faux, parce que le modèle apprendrait avec la réponse déjà présente.

### 9) Pourquoi sauvegarde-t-on `train.csv` et `test.csv` ?

On sauvegarde `train.csv` et `test.csv` pour garder une trace exacte des données utilisées.

C’est important pour la reproductibilité. Si plus tard on veut comprendre ou refaire une expérience, on peut retrouver les mêmes données.

Dans un projet MLOps, garder les données liées aux runs est important pour la traçabilité.

### 10) Pourquoi MLflow est utile dans un projet MLOps réel ?

MLflow est utile parce qu’il permet de suivre les expériences de façon organisée.

Dans un vrai projet, on peut avoir plusieurs modèles, plusieurs paramètres et beaucoup de résultats. Sans outil de tracking, on peut vite perdre le fil.

MLflow centralise les paramètres, les métriques, les modèles et les artefacts. Ça facilite aussi le travail en équipe.

### 11) Pourquoi un bon score sur le jeu de test ne suffit pas toujours pour déployer un modèle en production ?

Un bon score sur le jeu de test est important, mais ça ne suffit pas toujours.

En production, les données peuvent être différentes des données d’entraînement. Il peut aussi y avoir des valeurs manquantes, des nouvelles situations ou des changements dans le temps.

Il faut aussi vérifier d’autres points, comme la stabilité du modèle, le temps de réponse, la robustesse et les contraintes métier.

Donc avant de déployer, il faut faire plus de validations.

### 12) Que faudrait-il ajouter pour rendre ce projet plus professionnel ?

Pour rendre le projet plus professionnel, j’ajouterais Docker. Docker permettrait d’avoir le même environnement partout, peu importe la machine utilisée.

J’ajouterais aussi GitHub Actions pour automatiser certaines étapes, comme les tests ou l’exécution du script.

On pourrait aussi ajouter :

* un fichier `requirements.txt` ;
* des tests unitaires ;
* une meilleure documentation ;
* une validation croisée ;
* une meilleure gestion des logs ;
* un pipeline CI/CD plus complet.

Avec ces éléments, le projet serait plus proche d’un vrai projet MLOps utilisé en entreprise.
