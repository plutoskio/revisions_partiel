Voici un guide méthodologique clair, structuré sous forme de fichier Markdown, pour déterminer l'irréductibilité d'un polynôme dans n'importe quel contexte de ton examen. Tu peux l'utiliser comme une liste de contrôle (checklist) pour chaque question de l'exercice 2.

---

# 🌳 Arbre de Décision : Irréductibilité des Polynômes

Ce guide permet de choisir la méthode la plus rapide selon le corps $K$ et le degré $n$ du polynôme $P$.

---

## 1. Contexte : Corps de base ($K$)

### 🟦 Cas A : Dans $\mathbb{C}[X]$

* 
**Règle :** $P$ est irréductible **si et seulement si** $\deg(P) = 1$.


* *Action :* Si $\deg(P) > 1$, il est toujours réductible.

### 🟩 Cas B : Dans $\mathbb{R}[X]$

* **Règle :** $P$ est irréductible **si et seulement si** :
1. 
$\deg(P) = 1$.


2. 
$\deg(P) = 2$ **ET** le discriminant $\Delta < 0$.




* 
*Action :* Tout polynôme de degré $\geq 3$ est réductible dans $\mathbb{R}[X]$.



---

## 2. Cas C : Dans les Corps Finis $\mathbb{F}_p[X]$ (ex: $\mathbb{Z}/p\mathbb{Z}$)

### 🔸 Étape 1 : Vérifier les racines ($P(a) = 0$)

* Teste les valeurs $\{0, 1, \dots, p-1\}$ (utilise les négatifs comme $-1, -2$ pour simplifier).


* 
**Si $P$ a une racine :** Il est **Réductible** (divisible par $X-a$).



### 🔸 Étape 2 : Décider selon le degré ($n$)

* 
**Si $n = 1$ :** Toujours **Irréductible**.


* **Si $n = 2$ ou $3$ :**
* Aucune racine = **Irréductible**.




* **Si $n \geq 4$ :**
* Aucune racine $\neq$ Irréductible. Il faut vérifier s'il n'est pas le produit de deux polynômes de degrés inférieurs (ex: $2+2$ pour un degré 4).


* 
*Exemple sur $\mathbb{F}_2$ :* Si pas de racine, tester la division par $X^2+X+1$.





---

## 3. Cas D : Dans $\mathbb{Q}[X]$ ou $\mathbb{Z}[X]$ (Le plus fréquent)

### 🚀 Méthode 1 : Critère d'Eisenstein (Le réflexe n°1)

Chercher un nombre premier $p$ tel que :

1. 
$p$ divise tous les coefficients sauf le premier ($a_n$).


2. 
$p$ ne divise pas $a_n$.


3. 
$p^2$ ne divise pas le terme constant ($a_0$).



* 
**Résultat :** Si ces 3 conditions sont vraies, $P$ est **Irréductible** sur $\mathbb{Q}$.



### 🚀 Méthode 2 : Translation (Si Eisenstein échoue)

* Calculer $P(X+1)$ ou $P(X-1)$.


* Appliquer Eisenstein sur ce nouveau polynôme.


* 
**Résultat :** Si $P(X+1)$ est irréductible, alors $P(X)$ l'est aussi.



### 🚀 Méthode 3 : Réduction Modulo $p$

* Réduire les coefficients modulo un petit premier $p$ (souvent $p=2$).


* 
**Résultat :** Si $\overline{P}$ est irréductible dans $\mathbb{F}_p[X]$, alors $P$ est **Irréductible** dans $\mathbb{Z}[X]$ et $\mathbb{Q}[X]$.



### 🚀 Méthode 4 : Test des Racines Rationnelles (Pour $n=2, 3$)

* Lister les candidats $p/q$ où $p|a_0$ et $q|a_n$.


* Si aucun candidat n'est racine, le polynôme de degré 2 ou 3 est **Irréductible**.

