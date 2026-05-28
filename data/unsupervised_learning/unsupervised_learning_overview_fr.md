L'apprentissage non supervisé est différent de la plupart des méthodes de machine learning car il n'y a pas de "vérité terrain" (ground truth) ou de valeur cible pour comparer la sortie du modèle.

# Hard vs. Soft Clustering

**Hard Clustering (Partitionnement strict), ex. K-Means :** Frontières absolues. Chaque point de donnée appartient à exactement un cluster. Mathématiquement : $P(C_k|X_i) \in \{0,1\}$.

**Soft Clustering (Partitionnement flou), ex. GMM :** Frontières probabilistes. Les points de données ont une distribution de probabilité sur tous les clusters, par exemple 80% pour le Cluster A et 20% pour le Cluster B. Mathématiquement : $P(C_k|X_i) \in [0,1]$.

# Familles de Clustering

**1. Clustering basé sur les Centroïdes (Partitionnement)**
- **Comment ça marche :** Trouve un point central (centroïde) pour chaque cluster et regroupe les données selon la distance la plus courte à ces centres. Suppose des clusters sphériques et de taille approximativement similaire.
- **Algorithmes :** K-Means.
- **Affectation :** Stricte (Hard).

**2. Clustering basé sur les Distributions (Probabiliste)**
- **Comment ça marche :** Suppose que les données ont été générées par un mélange de distributions statistiques (ex : distributions Gaussiennes/Normales). Capture le centre, l'étalement et l'orientation (formes elliptiques) des données.
- **Algorithmes :** Gaussian Mixture Models (GMM).
- **Affectation :** Floue (Soft).

**3. Clustering Hiérarchique**
- **Comment ça marche :** Construit un arbre de clusters (un dendrogramme) en fusionnant itérativement de petits clusters ensemble (Ascendant / Agglomératif) ou en divisant un cluster géant (Descendant / Divisif). Ne nécessite pas de deviner le nombre de clusters à l'avance.
- **Algorithmes :** Agglomerative Clustering (utilisant des liens comme Ward, Single, Complete).
- **Affectation :** Stricte (Hard).

**4. Clustering basé sur la Densité**
- **Comment ça marche :** Recherche des régions continues où les points de données sont serrés les uns contre les autres, séparées par de l'espace vide. Excellent pour trouver des formes non-linéaires (comme le dataset "Moons") et signale automatiquement les points isolés comme du bruit ou des anomalies.
- **Algorithmes :** DBSCAN, HDBSCAN.
- **Affectation :** Stricte (Hard).

# Modèle K-Means

**Catégorie :** Clustering basé sur les Centroïdes (Partitionnement)

**Vidéo :** [K-Means explained](https://www.youtube.com/watch?v=4b5d3muPQmA&pp=ygUTc3RhdHMgcXVlc3Qgay1tZWFucw%3D%3D)

## Comment le modèle fonctionne

**Initialisation :** Vous commencez par définir le nombre de clusters que vous voulez identifier, ce qui représente le K. L'algorithme sélectionne ensuite aléatoirement K points de données distincts pour agir comme centres de clusters initiaux.

**Affectation :** L'algorithme mesure la distance de chaque point de donnée aux centres de clusters initiaux et affecte chaque point au cluster qui lui est le plus proche.

**Mise à jour :** Une fois tous les points groupés, il calcule la moyenne (mean) de chaque cluster nouvellement formé pour établir de nouveaux centres de clusters.

**Itération :** Il répète le processus de mesure des distances et de réaffectation des points aux valeurs moyennes les plus proches jusqu'à ce que les clusters ne changent plus.

## Les mathématiques derrière le modèle

K-means utilise la distance Euclidienne pour affecter les points au centre de cluster le plus proche : $d(x, c) = \sqrt{\sum_{j=1}^{p}(x_j - c_j)^2}$.

Parce qu'il dépend de la distance, les variables doivent être mises à l'échelle (scaled). Il souffre également du fléau de la dimensionnalité : en haute dimension, les distances deviennent moins informatives.

## Métriques utiles

K-means utilise couramment l'Inertie, aussi appelée WCSS, et le Score de Silhouette.

## Hyperparamètres K-Means (Scikit-learn)

**n_clusters ($K$) :** Le nombre de clusters que vous voulez que l'algorithme trouve, qui doit être prédéfini.

**n_init :** Le nombre de fois que l'algorithme s'exécute à partir de zéro avec différents points de départ aléatoires. Le modèle conserve finalement l'exécution qui produit l'Inertie la plus basse.

**max_iter :** Le nombre maximum d'étapes, ou d'itérations, qu'une seule exécution effectuera pour tenter d'atteindre la convergence avant d'être forcée de s'arrêter.

## Limitations

**Prédéfinir K :** K-means ne peut pas choisir le nombre optimal de clusters ; les utilisateurs doivent tester différentes valeurs de K.

**Sensibilité aux points de départ :** Des centres initiaux aléatoires peuvent conduire à de mauvais clusters, c'est pourquoi l'algorithme est généralement exécuté plusieurs fois et conserve le résultat avec la variance la plus faible.

## Biais

K-means a deux biais principaux :

1. L'hypothèse sphérique
2. L'hypothèse convexe

# Gaussian Mixture Models (GMM)

**Catégorie :** Clustering basé sur les Distributions (Probabiliste)

**Vidéo :** [Gaussian Mixture Models explained](https://www.youtube.com/watch?v=EWd1xRkyEog)

Une alternative probabiliste et générative à K-means. Il suppose que les données sont générées par un mélange de distributions Gaussiennes, permettant des formes de clusters plus flexibles.

## Comment le modèle fonctionne

**Clustering Flou (Soft) :** Les points sont affectés à des probabilités à travers les clusters, comme 80% Cluster A et 20% Cluster B.

**Optimisation (Algorithme EM) :** GMM utilise l'Expectation-Maximization pour trouver les Estimations de Maximum de Vraisemblance (MLE), en répétant deux étapes jusqu'à la convergence.

**E-Step (Expectation) :** Calcule la probabilité de chaque point, ou responsabilité, d'appartenir à chaque cluster.

**M-Step (Maximization) :** Met à jour les paramètres des clusters pour maximiser la vraisemblance des données.

## Les mathématiques derrière le modèle

GMM utilise la distance de Mahalanobis, qui mesure la distance tout en tenant compte de la covariance de chaque cluster, ainsi les clusters étirés ou tournés sont mieux gérés qu'avec la distance Euclidienne.

## Paramètres des Clusters

**Moyenne ($\mu_k$) :** Le centre de la Gaussienne.

**Matrice de Covariance ($\Sigma_k$) :** La forme, l'étalement et l'orientation.

**Poids du mélange ($\pi_k$) :** La probabilité a priori, permettant des clusters de tailles différentes.

## Hyperparamètres (Types de Covariance)

Le `covariance_type` contrôle la géométrie autorisée des clusters :

**spherical :** Formes circulaires avec des rayons différents.

**diag :** Formes elliptiques, mais les axes restent parallèles aux axes des variables.

**tied :** Tous les clusters partagent la même forme, taille et rotation.

**full :** Les clusters peuvent avoir n'importe quelle forme ellipsoïdale, taille et rotation.

## Limitations

**Complexité et Surapprentissage :** Plus lent et plus coûteux que K-means. Une covariance complète en haute dimension avec des données limitées présente un risque élevé de surapprentissage.

# Clustering Hiérarchique (Agglomératif)

**Catégorie :** Clustering Hiérarchique

## Comment le modèle fonctionne (Étape par Étape)

L'approche la plus courante est **Agglomérative (Ascendante)**, qui est un processus itératif et glouton :

1. **Départ :** Chaque point de donnée commence comme son propre cluster individuel (ex: $N$ points = $N$ clusters).
2. **Trouver :** L'algorithme calcule les distances entre tous les clusters existants et trouve les deux plus proches.
3. **Fusionner :** Il fusionne ces deux clusters les plus proches en un seul nouveau cluster.
4. **Mettre à jour :** Il recalcule les distances entre ce nouveau cluster plus grand et tous les autres clusters restants.
5. **Répéter :** Il répète les étapes 2 à 4 jusqu'à ce que tous les points de données aient été fusionnés en un seul cluster massif.

## Les mathématiques derrière le modèle

L'algorithme s'appuie sur deux niveaux distincts de calcul de distance :
1. **Métrique de Distance (Point-à-Point) :** Comment la distance entre les points individuels est mesurée (ex: Euclidienne ou Manhattan pour les données quantitatives, Hamming pour le qualitatif). Comme K-Means, cela nécessite une mise à l'échelle des variables.
2. **Critère de Liaison (Cluster-à-Cluster) :** Une fois les points groupés, cela définit comment la distance entre des *clusters entiers* est mesurée pour décider quels deux clusters fusionner ensuite.

## Hyperparamètres (Critères de Liaison)

* **Single Linkage (Lien Simple) :** Distance minimale entre n'importe quels deux points dans des clusters différents. Enclin à l'"Effet de Chaînage" (clusters longs, fins et connectés).
* **Complete Linkage (Lien Complet) :** Distance maximale entre les points dans des clusters différents. Encourage les clusters compacts et sphériques mais est hautement sensible aux outliers.
* **Average Linkage (Lien Moyen) :** Distance moyenne entre tous les points dans des clusters différents. Une approche équilibrée qui est moins affectée par les outliers.
* **Ward's Method (Méthode de Ward) :** Minimise l'augmentation de la variance intra-cluster lors de la fusion. Très similaire à l'inertie de K-Means, mais appliquée de manière hiérarchique. C'est souvent le choix optimal en pratique.

### Comparaison Visuelle : Single vs. Complete Linkage

| Métrique | Règle | Forme Résultante | Faiblesse Majeure |
| :--- | :--- | :--- | :--- |
| **Single Linkage** | **Distance Min** entre les deux points les plus *proches*. | Flexible, allongée (peut suivre des "lunes"). | **Effet de Chaînage :** Fusionne des groupes distincts si un seul point "pont" existe. |
| **Complete Linkage** | **Distance Max** entre les deux points les plus *éloignés*. | Compacte, serrée et sphérique. | **Sensibilité aux Outliers :** Un seul outlier peut maintenir des clusters logiques séparés. |

## Détermination de $K$ (Dendrogrammes)

Contrairement à K-Means, le nombre de clusters ($K$) est choisi **après** l'exécution du modèle en analysant un **Dendrogramme** (un diagramme en forme d'arbre enregistrant la séquence des fusions).
*   **Axe Y :** Représente la distance (ou la variance) à laquelle les clusters ont été fusionnés. Des sauts verticaux plus grands signifient que les clusters fusionnés étaient moins similaires.
*   **Couper l'Arbre :** Pour déterminer $K$, tracez une ligne de seuil horizontale à travers le dendrogramme. Le nombre de lignes verticales qu'elle intersecte est égal à $K$.
*   **Bonne Pratique :** Tracez la coupe à travers la plus grande distance verticale qui n'intersecte aucune ligne de fusion horizontale. Cela produit les clusters les plus distincts et naturels.

## Limitations

**Coûteux en calcul :** La complexité temporelle est élevée ($O(n^2)$ ou $O(n^3)$). Il ne peut pas passer à l'échelle pour des jeux de données massifs avec des millions de lignes.
**Algorithme Glouton :** Une fois qu'une fusion est effectuée, elle ne peut pas être annulée ou réévaluée plus tard.
**Sensibilité aux outliers :** Selon la méthode de liaison utilisée (en particulier le Lien Complet), les outliers peuvent gravement impacter les structures des clusters.

# DBSCAN (Density-Based Spatial Clustering)

**Catégorie :** Clustering basé sur la Densité

**Vidéo :** [DBSCAN Explained](https://www.youtube.com/watch?v=RDZUdRSDOok)

## Comment le modèle fonctionne (Étape par Étape)

Contrairement aux modèles de centroïdes qui forcent les données dans des sphères rigides, DBSCAN trouve des régions contiguës et arbitraires de haute densité :

1. **Choisir :** L'algorithme choisit un point arbitraire non visité et évalue la densité dans son rayon **Epsilon ($\epsilon$)**.
2. **Classer :**
    * **Core Point (Point Cœur) :** S'il a $\ge$ **MinPts** voisins, un nouveau cluster est démarré.
    * **Border Point (Point Frontière) :** S'il a peu de voisins mais se trouve dans le rayon d'un Point Cœur, il rejoint ce cluster.
    * **Noise Point (Point de Bruit) :** S'il n'est ni l'un ni l'autre, il est marqué comme un outlier (Bruit).
3. **Étendre :** Si un point est un Point Cœur, tous les voisins dans son rayon sont ajoutés au cluster. Si ces voisins sont *aussi* des Points Cœurs, leurs propres voisinages sont ajoutés au cluster (une réaction en chaîne).
4. **Répéter :** Le processus continue jusqu'à ce que le cluster soit complètement entouré de points frontières et de bruit, puis il passe au point non visité suivant.

## La Logique : Classification des Points

* **Points Cœurs :** Le "cœur dense" du cluster. Ils déclenchent l'expansion.
* **Points Frontières :** Les "bords" du cluster. Ils font partie du groupe mais n'ont pas assez de densité pour l'étendre davantage.
* **Points de Bruit :** Anomalies isolées qui n'appartiennent à aucun cluster.

## Hyperparamètres

DBSCAN ne nécessite pas de $K$ prédéfini, mais il est extrêmement sensible à :

* **Epsilon ($\epsilon$ / eps) :** Le rayon maximum utilisé pour chercher des voisins.
    * **Réglage :** Utilisez un **graphe des K-distances** ; le "coude" ou point d'inflexion indique l'optimal $\epsilon$.
* **MinPts (min_samples) :** Le nombre minimum de points nécessaires pour former un cluster dense.
    * **Règle empirique :** Généralement fixé au nombre de dimensions + 1 (ex: 4 ou 5 pour des données en 2D).

## Limitations & HDBSCAN

* **Densités Variables :** Le DBSCAN standard échoue si un seul jeu de données contient certains clusters très denses et d'autres très épars, car une seule valeur d'$\epsilon$ fixe ne peut pas capturer les deux de manière précise.
* **L'amélioration HDBSCAN :** Le DBSCAN Hiérarchique (État de l'art) résout ce problème en évaluant toutes les valeurs d'$\epsilon$ possibles simultanément, en extrayant les clusters basés sur la stabilité plutôt que sur un rayon fixe unique.
* **Fléau de la dimensionnalité :** La précision se dégrade rapidement dans les espaces de très haute dimension car la distance Euclidienne devient moins significative.

# Matrice de Décision Résumée : Clustering

| Modèle | Catégorie | Scalabilité | Points Forts | Faiblesses |
| :--- | :--- | :--- | :--- | :--- |
| **K-Means** | Centroïde | **Haute** $O(nkdi)$ | Rapide, simple, scalable. | Nécessite $K$, biais sphérique, sensible aux outliers. |
| **GMM** | Distribution | **Modérée** $O(nkdi)$ | Formes flexibles (ellipses), affectation floue. | Nécessite $K$, sensible à l'initialisation, convergence lente. |
| **Hiérarchique** | Hiérarchie | **Faible** $O(n^2)$ | Interprétable (Dendrogramme), pas de $K$ prédéfini. | Lent sur gros volumes, glouton (ne peut pas annuler les fusions). |
| **DBSCAN** | Densité | **Modérée** $O(n \log n)$ | Formes complexes, détection du bruit, pas de $K$. | Sensible à $\epsilon$ et aux variations de densité. |

# Métriques

## Inertie (WCSS) et le Graphe du Coude (Elbow Plot)

L'inertie mesure la variance totale intra-cluster. Elle diminue à mesure que $K$ augmente.

**Elbow Plot :** Choisissez le $K$ où l'inertie s'arrête de chuter brusquement.

## Le Score de Silhouette

Note les clusters de -1 à 1 ; plus on est proche de 1, mieux c'est.

Mesure la cohésion au sein des clusters et la séparation entre les clusters. Des scores négatifs suggèrent une mauvaise affectation.

---

# PARTIE 2 : RÉDUCTION DE DIMENSIONNALITÉ

## Le Fléau de la Dimensionnalité

À mesure que le nombre de variables (dimensions) dans un jeu de données augmente, le volume de l'espace des variables croît de manière exponentielle. Cela cause plusieurs problèmes :
* **Sparsité des données :** Les points de données s'isolent, faisant perdre leur sens aux métriques de distance (comme la distance Euclidienne).
* **Surapprentissage (Overfitting) :** Les modèles deviennent trop complexes, augmentant le coût de calcul et le risque de mémoriser le bruit.
* **Multicolinéarité :** Les variables sont fortement corrélées, ce qui déstabilise les modèles prédictifs.

Pour résoudre cela, nous utilisons l'Extraction de Variables / Réduction de Dimensionnalité pour créer moins de nouvelles variables qui capturent l'essentiel de l'information.

## 1. Analyse en Composantes Principales (PCA) - *Linéaire*

**Catégorie :** Réduction de Dimensionnalité Linéaire

**Objectif :** Transformer un grand ensemble de variables corrélées en un plus petit ensemble de variables *non corrélées* (Composantes Principales) tout en conservant autant de variance originale (information) que possible.

### Fonctionnement (Les Mathématiques)
1. **Covariance :** Analyse comment toutes les variables évoluent ensemble.
2. **Vecteurs Propres (Directions) :** Trouve de nouveaux axes orthogonaux. Le premier axe (**PC1**) est tracé dans la direction qui maximise l'étalement (variance) des données. PC2 est tracé orthogonalement à PC1, capturant la deuxième plus grande variance, et ainsi de suite.
3. **Valeurs Propres (Magnitude) :** Donne un score à chaque nouvel axe indiquant quelle proportion de la variance totale il explique.
4. **L'Astuce SVD :** En pratique (ex: Scikit-Learn), la PCA utilise la **Décomposition en Valeurs Singulières (SVD)** pour calculer ces composantes instantanément sans avoir besoin de la matrice de covariance lourde en calcul.

### Étape Cruciale : Prétraitement
**La Standardisation est Obligatoire :** La PCA est un exercice de maximisation de la variance. Si la Variable A est en millions (ex: Revenus) et la Variable B en décimales (ex: Taux de conversion), la PCA pensera à tort que la Variable A est plus importante juste parce que ses chiffres sont plus grands. **Appliquez toujours un `StandardScaler` avant de fitter une PCA.**

### Interprétation & Visualisations
* **Scree Plot (Choix des composantes) :** Un graphique linéaire montrant le pourcentage de variance expliqué par chaque composante principale.
    * **Méthode du Coude :** Cherchez une chute abrupte suivie d'un plateau (le "coude") pour trouver le nombre optimal de composantes.
    * **Règle de la Variance Cumulée :** Dans les contextes business, on garde assez de composantes pour expliquer 70% à 80% de la variance totale.
* **Cercle de Corrélation :** Visualise comment les variables originales sont liées aux nouvelles Composantes Principales.
    * Flèches proches = corrélées positivement. Opposées = corrélées négativement. Orthogonales (90°) = non corrélées.
    * Plus une flèche est proche du bord du cercle, mieux elle est représentée par ce plan 2D.
* **Biplot :** Combine le Cercle de Corrélation (variables/flèches) et les Projections d'Échantillons (points de données/points) sur le même graphique. Cela permet de voir *pourquoi* certains points se regroupent en fonction des variables originales.

### Applications Business
1. **Création de KPIs Composites :** Synthétiser 15 indicateurs complexes en un seul "Score Global" (PC1).
2. **Lutte contre la Multicolinéarité :** Injecter des Composantes Principales strictement orthogonales (corrélation 0) dans des modèles de régression au lieu de données brutes fortement corrélées, rendant le modèle très stable.
3. **Prétraitement pour le Clustering :** Réduire les dimensions (ex: 200 variables $\rightarrow$ 10 PCs) avant de lancer un K-Means pour vaincre le fléau de la dimensionnalité et supprimer le bruit.
4. **Détection d'Anomalies / Fraude :** Si une transaction projetée sur l'espace PCA atterrit extrêmement loin du centre (origine), elle viole les règles de corrélation standards et est probablement anormale.

### Limitation Majeure
**Hypothèse Linéaire :** La PCA a un défaut majeur : elle suppose que toutes les relations sont linéaires. Si vos données reposent sur une variété (manifold) complexe, tordue et non-linéaire (comme un "Swiss Roll"), la PCA échouera à capturer la structure réelle.

## 2. Manifold Learning - *Non-Linéaire*

**Objectif :** Au-delà des projections linéaires comme la PCA, ces modèles utilisent des "Variétés" (Manifolds) pour projeter des formes non-linéaires et complexes dans un espace de dimension inférieure.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
**Catégorie :** Cartographie de Proximité Locale Non-Linéaire

*   **Objectif Central :** Se concentre strictement sur la préservation des **Voisinages Locaux**. Il calcule la probabilité que deux points soient voisins dans l'espace de haute dimension et tente de faire correspondre cette probabilité exacte dans la projection 2D/3D.
*   **Le "Problème d'Écrasement" (Crowding Problem) :** L'espace de haute dimension a un "volume" nettement plus important que la 2D. En écrasant les données en deux dimensions, les points modérément distants en haute dimension ont naturellement tendance à s'entasser, créant des amas qui se chevauchent.
*   **La Solution (Loi de Student à queues lourdes) :** t-SNE utilise une loi de Student (t) dans l'espace de projection. Parce qu'elle a des **queues lourdes**, elle exerce une force de répulsion plus forte sur les points distants, repoussant les clusters pour les laisser "respirer" et rendant les espaces entre eux visibles.
*   **Perplexité :** Le principal paramètre de réglage qui définit le nombre de voisins effectifs que chaque point considère.
    *   **Faible Perplexité :** Priorise les micro-clusters et les motifs locaux très fins.
    *   **Haute Perplexité :** Capture la structure globale et macroscopique des données.
*   **Mises en garde critiques sur l'interprétation :**
    *   **Les distances globales ne sont pas fiables :** L'écart physique entre le Cluster A et le Cluster B sur votre écran ne représente pas fidèlement leur similitude globale.
    *   **Non-Déterministe :** L'algorithme utilisant une initialisation aléatoire, les clusters peuvent apparaître à des positions ou orientations différentes à chaque exécution.
    *   **La densité n'est pas préservée :** Un cluster "dense" en 2D ne signifie pas forcément que les points étaient plus serrés en haute dimension.

### UMAP (Uniform Manifold Approximation and Projection)
**Catégorie :** Apprentissage de Variété Topologique Non-Linéaire

*   **Objectif Central :** Se concentre sur la **Préservation de la Structure Topologique**. Au lieu de simplement regarder les probabilités de voisinage, il construit un "squelette" mathématique (un graphe) de l'ensemble du jeu de données. Il suppose que les données sont uniformément distribuées sur une surface (variété) et tente de "dérouler" cette surface dans un espace de dimension inférieure.
*   **L'Avantage Global :** Alors que t-SNE déchire souvent les données pour mettre en avant les clusters locaux, UMAP maintient la distance relative entre ces clusters. Cela signifie que la **géométrie globale** macroscopique de vos données est préservée (ex: si le Cluster A est conceptuellement "plus proche" du Cluster B que du Cluster C, UMAP le montrera probablement).
*   **Performance & Polyvalence :** 
    *   **Grande Vitesse :** Nettement plus efficace en calcul que t-SNE ; il peut traiter des millions de lignes sans sous-échantillonnage massif.
    *   **d-Dimensionnalité :** Contrairement au t-SNE (qui sert surtout à la visualisation 2D/3D), UMAP peut réduire les données vers *n'importe quel* nombre de dimensions, ce qui en fait une étape de prétraitement puissante pour d'autres modèles de machine learning.
*   **Hyperparamètres :**
    *   **n_neighbors :** Contrôle le "niveau de zoom" du graphe topologique. 
        *   Les valeurs faibles se concentrent sur les structures locales fines (les arbres individuels). 
        *   Les valeurs élevées capturent la forme globale (la forêt entière).
    *   **min_dist :** Contrôle la compacité avec laquelle l'algorithme regroupe les points dans la projection. 
        *   Les valeurs faibles créent des clusters denses et agglomérés (utile pour trouver des groupes distincts).
        *   Les valeurs élevées permettent aux points de s'étaler (utile pour voir la continuité et le "flux" des données).

# Matrice de Décision Résumée : Réduction de Dimensionnalité

| Modèle | Approche | Scalabilité | Objectif Principal | Idéal pour |
| :--- | :--- | :--- | :--- | :--- |
| **PCA** | Linéaire (Global) | **Haute** | Maximisation de la Variance | Réduction de variables, KPIs composites, Multi-colinéarité. |
| **t-SNE** | Non-Linéaire (Local) | **Faible** | Séparabilité des Clusters | **Visualisation 2D/3D uniquement.** |
| **UMAP** | Non-Linéaire (Topologique)| **Haute** | Préservation de la Structure | Réduction rapide pour Big Data, Prétraitement pour le clustering. |

---

# PARTIE 3 : DÉTECTION D'ANOMALIES

## Le Contexte Business & Le Coût Réel
La détection d'anomalies se concentre sur la recherche de "l'aiguille dans la botte de foin" (souvent <1% des données). Comme ces événements sont très rares, l'exactitude (accuracy) classique est inutile. Au lieu de cela, les modèles sont réglés en fonction du risque financier asymétrique :
*   **Le Coût d'un Faux Positif (FP) :** Le modèle signale un événement normal. *Impact :* Temps d'enquête perdu, "Lassitude face aux alertes" (Alert Fatigue), ou blocage de la carte de crédit d'un client légitime (risque de désabonnement).
*   **Le Coût d'un Faux Négatif (FN) :** Le modèle rate une anomalie réelle. *Impact :* Une machine tombe en panne arrêtant la production (100k$/heure), ou une fraude massive n'est pas détectée.
*   **Paramètre de Contamination :** Le seuil que nous fixons pour équilibrer FP et FN en fonction du risque le plus dangereux pour l'entreprise.

## Les Deux Paradigmes & L'Effet de Masquage
1. **Détection d'Outliers (La Réalité Bordélique) :** Non supervisée. Le jeu de données d'entraînement est "pollué" par des anomalies inconnues. Le but est de fitter le mode central des données et d'isoler les points éloignés.
2. **Détection de Nouveauté (Le Cas Idéal) :** Semi-supervisée. L'entraînement se fait sur des données strictement *propres/normales* (ex: 100 heures d'un moteur d'avion parfaitement sain). Le modèle apprend cet état exact et signale toute déviation future.
*   **L'Effet de Masquage (Masking Effect) :** Pourquoi ne pas simplement utiliser les mathématiques standards (Moyenne et Covariance) ? Parce que les anomalies ont une force gravitationnelle massive. Elles tirent le "centre" calculé vers elles, masquant leur propre nature anormale et provoquant des Faux Négatifs. Cela nécessite des algorithmes robustes.

## Modèles Clés

### 1. One-Class SVM (Basé sur les Frontières)
**Catégorie :** Enveloppe Géométrique / Détection de Nouveauté
*   **Intuition (Le Film Étirable) :** Imaginez envelopper vos données normales et denses dans un film étirable serré. Tout ce qui tombe en dehors de cette enveloppe plastique est marqué comme une anomalie.
*   **Le Mécanisme :** Utilise le **Noyau RBF** pour projeter les données dans un espace de dimension infinie, ce qui lui permet de tracer des contours complexes et non-linéaires qui épousent étroitement les données normales, quelle que soit leur forme.
*   **Hyperparamètre - $\nu$ (Nu) :** Représente votre "marge de tolérance" pour les anomalies dans le jeu d'entraînement.
    *   **Faible $\nu$ (ex: 0.01) :** Crée une frontière large et confiante. Suppose que vos données d'entraînement sont propres à 99%.
    *   **Haute $\nu$ (ex: 0.10) :** Crée une frontière plus serrée. Exclut activement les 10% extérieurs de vos données d'entraînement pour éviter le surapprentissage au bruit.

### 2. Isolation Forest (Basé sur l'Isolation) - *SOTA pour le Big Data*
**Catégorie :** Partitionnement Aléatoire / Détection d'Outliers
*   **Intuition (Un Changement de Paradigme) :** Au lieu d'essayer de définir ce qui est "normal", iForest cible activement les anomalies. Comme les anomalies sont **"Rares et Différentes"**, elles sont beaucoup plus faciles à isoler du reste de la meute.
*   **Le Mécanisme (Coupes Aléatoires) :** L'algorithme effectue des "coupes" (splits) verticales et horizontales aléatoires à travers les données.
    *   **Points Normaux :** Sont enfouis profondément dans des clusters denses. Il faut *beaucoup* de coupes aléatoires pour réussir à isoler un seul point normal. (Longueur de chemin longue).
    *   **Anomalies :** Se trouvent dans le vide. Il ne faut généralement que 1 ou 2 coupes aléatoires pour les isoler. (Longueur de chemin courte).
*   **Points Forts :** Extrêmement rapide ($O(n \log n)$), gère magnifiquement les données de haute dimension et ne repose pas sur des calculs de distance complexes, ce qui en fait le standard de l'industrie pour les jeux de données massifs (comme le trafic Web ou les Logs).

### 3. Local Outlier Factor (LOF) (Basé sur la Proximité)
**Catégorie :** Comparaison de Densité / Anomalies Contextuelles
*   **Intuition :** Les modèles globaux (comme iForest) ratent souvent les anomalies "Locales" — des points qui semblent normaux globalement, mais qui sont anormaux par rapport à leur voisinage local spécifique et dense.
*   **Le Mécanisme :** LOF compare la densité d'un point $A$ à la densité de ses $k$ plus proches voisins.
    *   Si la densité de $A$ est similaire à celle de ses voisins $\rightarrow$ $LOF \approx 1$ (**Normal**).
    *   Si les voisins sont très serrés, mais que $A$ est un peu à l'écart $\rightarrow$ $LOF \gg 1$ (**Anomalie**).
*   **Idéal pour :** La fraude contextuelle et les anomalies géospatiales où le contexte local est primordial.

# Matrice de Décision Résumée : Détection d'Anomalies

| Modèle | Mécanisme Central | Meilleur Usage Business | Scalabilité |
| :--- | :--- | :--- | :--- |
| **Isolation Forest** | Coupes Aléatoires | Logs massifs, détection de Bots, Trafic Web | **Excellente** $O(n \log n)$ |
| **One-Class SVM** | Enveloppe (RBF) | Maintenance Prédictive (Données propres) | **Faible** $O(n^2)$ |
| **LOF** | Comparaison de Densité | Fraude contextuelle, Anomalies locales | **Modérée à Faible** |
