# patrick_machine_learning_french


> Vous trouverez ici les exemples des outils de que j'utilise en data science.Ce sont des notamment des algorithmes de machine learning algorithms implementé en **Python**.


## apprentissage supervisé

Dans l'apprentissage supervisé, on a un ensemble d'exemple en entrée (données d'apprentissage) qui ont des caractéristiques (caractéristiques=features en anglais) et qui sont etiquetées (etiquette=label en anglais).
Le but est de mettre en place un modèle capable à partir des features des données d'apprentissage, de prédire les lablels pour de nouveaux exemples.

### Regression

Dans les tâches de regression les valeurs prédites sont des nombres reels. 
Dans la pratique on cherche la droite, le plan, l'herperplan qui colle le mieux aux données.
_Exemple d'utilisations: prévision du prix des actions, analyse des ventes, dépendance d'une cause quelconque, etc._

####  Regression Linéaire 

_Exemple ci-dessous: predire le salaire `Salary`  en fonction du nombre d'année d'expérience `YearsExperience`_
<br>Dataset: [Salary_Data.csv](dataset/Salary_Data.csv) 
-  [Code | Regression Linéaire](supervised_learning/regression/simple_linear_regression.py) 

####  Regression Linéaire régularisée

_Exemple ci-dessous: prédire la quantité d’expression de l’antigène qui est associée à la détection du cancer de la prostate_
<br>Dataset: [TP_1_prostate_dataset.txt](dataset/TP_1_prostate_dataset.txt) 
-  [Code | Regression Linéaire régularisée LASSO](supervised_learning/regression/Regression_regularized_lasso.py)
-  [Code | Regression Linéaire régularisée RIDGE](supervised_learning/regression/Regression_regularized_ridge.py)

####  Regression multi-Linéaire

_Exemple : Pour prédire si il est sûr d'investir dans un startup en particulier ou non_
<br>Dataset: [50_Startups.csv](dataset/50_Startups.csv) 
-  [Code | Regression multi-Linéaire](supervised_learning/regression/multiple_linear_regression.py)

####  autres methodes non ensemblistes

_Exemple ci-dessous: predire le salaire `Salary`  en fonction du nombre d'année d'expérience `YearsExperience`_
<br>Dataset: [Position_Salaries.csv](dataset/Position_Salaries.csv) 
-  [Code | Machine à vecteurs de support (Regression) |Support Vector Regression (SVR)](supervised_learning/regression/svr.py) 

####  methodes ensemblistes

_Exemple ci-dessous: predire le salaire `Salary`  en fonction du nombre d'année d'expérience `YearsExperience`_
<br>Dataset: [Position_Salaries.csv](dataset/Position_Salaries.csv) 
-  [Code | Arbre de décision|Decision tree learning en anglais](ensemble_learning/boosting/decision_tree_regression.py) 
-  [Code | Forêt d'arbres décisionnels|Random forest en anglais](ensemble_learning/bagging/random_forest_regression.py)

### Classification

Dans les tâches de classification, on cherche distinguer les exemples d'entrée en fonction de certaines caractéristiques.
Les valeurs prédites sont les classes des exemples.

_Exemple d'utilisations: filtres anti-spam, détection de la langue, recherche de documents similaires, reconnaissance de lettres manuscrites, etc._

#### Exemple ci-dessous: L'objectif est de prévoir le salaire offert au nouvel employé en fonction de son niveau actuel.
<br>Dataset: [Social_Network_Ads.csv](dataset/Social_Network_Ads.csv) 

-  [Code |Régression logistique | Logistic Regression](supervised_learning/classification/logistic_regression.py)
-  [Code |Méthode des k plus proches voisins| K-NN ](supervised_learning/classification/knn.py) 
-  [Code | Machine à vecteurs de support|SVM ](supervised_learning/classification/svm.py) 
-  [Code | SVM à noyaux|Kernel method ](supervised_learning/classification/kernel_svm.py) 
-  [Code | Classification naïve bayésienne|naive bayes ](supervised_learning/classification/naive_bayes.py) 
-  [Code | Arbre de décision|Decision tree learning ](ensemble_learning/boosting/decision_tree_classification.py) 
-  [Code | Forêt d'arbres décisionnels|Random forest ](ensemble_learning/bagging/random_forest_classification.py)

## apprentissage non supervisé

L'apprentissage non supervisé cherche aussi à faire des prédiction à partir des exemples donnés.
Mais cette fois ces exemples ne sont pas labelisés: le modèle doit identifier les features seuls sans le travail préalable d'un expert.

### Clustering

Dans les tâches de Clustering, le modèle modèle apprend à sbdiviser les données qu'ont lui fournies selon des caractéristiques inconnues au préalable. 
L'algorithme lui-même décide quelle caractéristique utiliser pour le fractionnement.

_Exemples d'utilisation: segmentation du marché, analyse de réseaux sociaux, organisation de clusters informatiques, analyse de données astronomiques, compression d'images, etc._

#### K-means Algorithm

_Exemple ci-dessous: identifier des groupes de consommateurs_
<br>Dataset: [Mall_Customers.csv](dataset/Mall_Customers.csv) 
-  [Code |K-means Algorithm](unsupervised_learning/Clustering/kmeans.py) 

#### réduction de la dimension de des données

_Exemple ci-dessous: analyse en composantes principales de vins_
<br>Dataset: [Wine.csv](dataset/Wine.csv) 

-  [Code |analyse en composantes principales ACP (Algorithme)|PCA ](unsupervised_learning/dimensionality_Reduction/pca.py) 
-  [Code |analyse en composantes principales ACP avec noyau (Algorithme)|Kernel PCA ](unsupervised_learning/dimensionality_Reduction/lda.py) 
-  [Code |Analyse discriminante linéaire|LDA  (Algorithme)| Algorithm](unsupervised_learning/dimensionality_Reduction/kernel_pca.py) 


## Deep learning (Reseaux de neurones)

Le Deep learning est un type d'apprentissage automatique qui est développé dans le cadre des Le réseau neuronaux(ANN = Artificial Neural Networks).
Un réseau neuronal lui-même n'est pas un algorithme, mais plutôt un cadre permettant à de nombreux algorithmes d'apprentissage automatique de travailler ensemble et de traiter des entrées de données complexes.

_Exemples d'utilisation: alternative aux autres algorithmes classique, reconnaissance d'image, reconnaissance vocale, traduction de langue, etc._

### apprentissage supervisé

#### Au cour de tout les reseux neuronaux:la Rétropropagation du gradient | backpropagation
<br>Dataset: banque d'images de chiffres écrits à la main (MNIST). Utiliser cette [fonction](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py) développé by Michael Nielsen pour charger ces données.
-  [Code | network_nielsen](deep_learning/network_nielsen_redesigned.py) - J'ai remanié le code de Michael Nielsen.Un code **sans tensorflow,sans keras** pour bien suivre comprendre les reseaux neuronaux,notamment **l'algorithme du gradient stochastique et la backpropagation**.
-  [Code | tester le code remanié](deep_learning/test_network_nielsen_redesigned.py) - reconnaîtres des chiffres écrits à la main avec la descente de gradient stochastique via le code remanié.

#### Perceptron multicouche (MLP= Multilayer perceptron)
<br>Dataset: [Churn_Modelling.csv](deep_learning/ann.py) 
-  [Code | MLP](deep_learning/ann.py) - prédire si un client suivant va ou non quitter la banque dans les 6 mois.

#### Réseau neuronal convolutif (CNN= Convolutional Neural Networks)
<br>Dataset: [Salary_Data.csv](deep_learning/cnn.py) 
-  [Code | CNN](deep_learning/cnn.py) - prédire si l'animal pris en photo est un chien ou un chat.

#### Réseau de neurones récurrents (RNN = Recurrent Neural Networks)
<br>Datasets: [Google_Stock_Price_Test.csv](dataset/Google_Stock_Price_Test.csv) et [Google_Stock_Price_Train.csv](dataset/Google_Stock_Price_Train.csv) 
-  [Code | RNN](deep_learning/rnn.py) - prédire la valeur de l'action Google dans le futur.

### apprentissage non supervisé

#### Carte auto adaptative (SOM = Self Organizing Maps)
<br>Dataset: [Credit_Card_Applications.csv](supervised_learning/regression/simple_linear_regression.py) 
-  [Code | SOM](deep_learning/som.py) - détecter le potentiel de fraude d'un client à partir de des réponses à un formulaire.


#### Machine de Boltzmann (BM = Boltzmann Machines)
<br>Datasets: [ml-100k](deep_learning/rbm.py) et [ml-1m](supervised_learning/regression/simple_linear_regression.py) 
-  [Code | BM](deep_learning/rbm.py) - construire systeme de recommandation:prédire si un utilisateur va aimer ou non un film

#### Auto-encodeur (AE = AutoEncoders)
<br>Datasets: [ml-100k](supervised_learning/regression/simple_linear_regression.py) et [ml-1m](supervised_learning/regression/simple_linear_regression.py) 
-  [Code | AE](deep_learning/ae.py) - construire systeme de recommandation:  prédire la note de 1 à 5 qu'un utilisateur va donner à un film.



## Intelligence artificielle Map

![Machine Learning Map](machine-learning-map.png)

La source de cette mind-map sur l’apprentissage automatique est [sur article ce blog](https://vas3k.ru/blog/machine_learning/)

## Datasets

La liste des jeux de données utilisés se trouve dans le [dossier des datasets](dataset).
