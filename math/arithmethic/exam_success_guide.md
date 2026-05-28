# Arithmetique II - Guide pour reussir l'examen

Sources lues: `arithmetic_notes.pdf`, `arithmetic_td.pdf`, `arithmetic_solutions.pdf`, `arithmetic_practice_exam.pdf`, `arithmetic_exam.pdf`, `arithmetic_exam_solution.pdf`, `racine_decision.md`.

Objectif: savoir reconnaitre les structures algebriques, faire les calculs de polynomes, prouver l'irreductibilite, et manipuler les petits corps finis.  
Note: je saute volontairement la theorie generale des anneaux quotients, sauf le strict minimum operationnel pour les corps finis car c'est tombe dans les sujets.

---

## 1. Structures a connaitre parfaitement

### Groupe

Un groupe est un ensemble non vide `G` muni d'une loi interne `*` tel que:

1. Stabilite: pour tous `x,y in G`, `x*y in G`.
2. Associativite: `(x*y)*z = x*(y*z)`.
3. Element neutre: il existe `e in G` tel que `x*e = e*x = x`.
4. Inverse: pour tout `x in G`, il existe `y in G` tel que `x*y = y*x = e`.

Il est abelien si `x*y = y*x` pour tous `x,y`.

Exemples importants:

- `(Z/nZ,+)` est toujours un groupe abelien.
- `(Z/nZ)^x` muni de la multiplication est un groupe.
- `(Z/nZ \ {0}, x)` n'est pas forcement un groupe.
- Un espace vectoriel est un groupe abelien pour l'addition.

Proprietes a savoir rediger:

- L'element neutre est unique.
- L'inverse d'un element est unique.

Schema de preuve de l'unicite du neutre:

```text
Si e et e' sont deux neutres, alors e = e*e' = e'.
```

Schema de preuve de l'unicite de l'inverse:

```text
Si y et z sont deux inverses de x:
y = y*e = y*(x*z) = (y*x)*z = e*z = z.
```

### Sous-groupe

Soit `G` un groupe. Une partie non vide `H` de `G` est un sous-groupe si `H` est un groupe pour la loi de `G`.

Critere pratique:

```text
H < G  <=>  H non vide et pour tous x,y in H, x*y^{-1} in H.
```

En notation additive:

```text
H < G  <=>  H non vide et pour tous x,y in H, x-y in H.
```

Exemples types:

- `2Z` est un sous-groupe de `(Z,+)`.
- `2Z+1` n'est pas un sous-groupe de `(Z,+)` car il ne contient pas `0`.
- `Q^*` est un sous-groupe de `(R^*,x)`.
- L'intersection de deux sous-groupes est un sous-groupe.
- L'union de deux sous-groupes n'est pas toujours un sous-groupe.

Resultat utile:

```text
Tout sous-groupe additif de Z est de la forme nZ.
```

### Anneau

Un anneau est un ensemble `A` muni de deux lois `+` et `x` tel que:

1. `(A,+)` est un groupe abelien.
2. La multiplication est associative.
3. La multiplication est distributive par rapport a l'addition:

```text
x(y+z) = xy+xz
(x+y)z = xz+yz
```

Vocabulaire:

- Unitaire: il existe un element neutre multiplicatif `1_A`.
- Commutatif: `xy = yx` pour tous `x,y`.
- Integre: `xy = 0 => x = 0 ou y = 0`.
- `A^x`: ensemble des elements inversibles pour la multiplication.

Exemples importants:

- `Z` est un anneau commutatif, unitaire, integre.
- `Z/nZ` est un anneau commutatif unitaire.
- `Z/nZ` est integre si et seulement si `n` est premier.
- `2Z` est un anneau commutatif integre mais non unitaire.
- `M_n(R)` est un anneau unitaire, non commutatif, et non integre si `n >= 2`.

### Sous-anneau

Soit `A` un anneau. Une partie non vide `B` de `A` est un sous-anneau si elle est stable pour les operations de `A`.

Critere pratique:

```text
B sous-anneau de A
<=> B non vide et pour tous x,y in B:
    x-y in B
    xy in B
```

Ce critere revient souvent dans les exercices.

### Ideal

Soit `A` un anneau. Une partie `I` de `A` est un ideal si:

1. `(I,+)` est un sous-groupe de `(A,+)`.
2. Absorbance: pour tous `a in A` et `x in I`, on a `ax in I` et `xa in I`.

Exemples:

- `nZ` est un ideal de `Z`.
- Dans `K[X]`, l'ensemble des multiples d'un polynome fixe est un ideal.
- Dans un corps `K`, les seuls ideaux sont `{0}` et `K`.

Ideal engendre:

```text
<a> = {ax | x in A}
```

Dans `Z`:

```text
<n> = nZ
<n_1,...,n_k> = PGCD(n_1,...,n_k)Z
```

### Corps

Un corps est un anneau unitaire non nul dans lequel tout element non nul est inversible:

```text
K est un corps  <=>  K^x = K \ {0}.
```

Exemples:

- `Q`, `R`, `C` sont des corps.
- `Q[sqrt(2)] = {a+b sqrt(2) | a,b in Q}` est un corps.
- `Z/nZ` est un corps si et seulement si `n` est premier.

Propriete fondamentale:

```text
Tout corps est integre.
```

Preuve type:

```text
Si xy = 0 et x != 0, alors x est inversible.
Donc x^{-1}xy = x^{-1}0, donc y = 0.
```

---

## 2. Polynomes: base de calcul

### Polynome, degre, coefficient dominant

Dans un anneau commutatif unitaire integre `A`, un polynome est:

```text
P(X) = a_0 + a_1 X + ... + a_n X^n
```

Si `P != 0`, `deg(P)` est le plus grand indice `i` tel que `a_i != 0`.  
Par convention, `deg(0) = -infty`.

Un polynome est unitaire si son coefficient dominant vaut `1`.

Regles:

```text
deg(P+Q) <= max(deg P, deg Q)
deg(PQ) = deg P + deg Q    si l'anneau est integre
```

### L'anneau K[X]

Si `K` est un corps, alors `K[X]` est un anneau commutatif unitaire integre.

Pour repondre a une question comme celle du CC:

```text
K[X] est un anneau car:
- (K[X],+) est un groupe abelien;
- la multiplication est associative;
- la multiplication est distributive sur l'addition;
- la multiplication est commutative;
- le polynome constant 1 est neutre multiplicatif;
- K[X] est integre car le produit de deux polynomes non nuls a un coefficient dominant non nul.
```

Mais `K[X]` n'est pas un corps. Exemple:

```text
X n'est pas inversible dans K[X].
Si XQ = 1, alors deg(XQ) = 1+deg(Q) >= 1, contradiction avec deg(1)=0.
```

### Division euclidienne

Dans `K[X]`, si `K` est un corps et `B != 0`, alors:

```text
Il existe un unique couple (Q,R) tel que A = BQ + R et deg(R) < deg(B).
```

Methode:

1. Diviser les termes dominants.
2. Soustraire.
3. Recommencer jusqu'a ce que le degre du reste soit strictement plus petit.

Attention: la division euclidienne marche proprement dans `K[X]`, donc souvent dans `Q[X]`, `R[X]`, `F_p[X]`.

### PGCD et algorithme d'Euclide

Le PGCD de `A` et `B` dans `K[X]` est le polynome unitaire `D` tel que:

```text
D | A et D | B
Tout diviseur commun de A et B divise D
```

Algorithme:

```text
R_0 = A
R_1 = B
R_{n+1} = reste de la division de R_{n-1} par R_n
```

Le PGCD est le dernier reste non nul, rendu unitaire.

Astuce de calcul:

```text
Dans Q[X], on peut remplacer un reste non nul par un multiple non nul plus simple,
par exemple son normalise unitaire.
```

### Bezout

Pour `A,B in K[X]` non nuls:

```text
Il existe U,V in K[X] tels que AU + BV = PGCD(A,B).
```

En particulier:

```text
A et B sont premiers entre eux
<=> il existe U,V tels que AU + BV = 1.
```

Methode pour trouver `U,V`:

1. Faire Euclide.
2. Garder les egalites de restes.
3. Remonter les egalites depuis le dernier reste non nul.

---

## 3. Racines et irreductibilite

### Racine

Soit `P in K[X]` et `alpha in K`.

```text
alpha racine de P <=> P(alpha)=0
```

Propriete capitale:

```text
alpha racine de P <=> (X-alpha) divise P.
```

Un polynome non nul de degre `n` a au plus `n` racines dans un corps.

### Racines rationnelles

Soit:

```text
P(X) = a_n X^n + ... + a_0 in Z[X],
a_n != 0, a_0 != 0.
```

Si `p/q` est une racine rationnelle sous forme irreductible, alors:

```text
p | a_0
q | a_n
```

Cas utile:

- Si le polynome est unitaire, les racines rationnelles possibles sont les diviseurs entiers du terme constant.
- Pour un degre 2 ou 3 dans `Q[X]`, aucune racine rationnelle implique irreductible.

### Polynome irreductible

Soit `A` un anneau integre. Un polynome `P in A[X]` est irreductible si:

```text
deg(P) >= 1
et si P = UV, alors U est inversible ou V est inversible.
```

Autrement dit: on ne peut pas le factoriser en deux polynomes de degres strictement positifs.

Attention:

```text
Si deg(P) >= 2 et P a une racine dans A, alors P est reductible.
```

Mais l'inverse est faux en degre 4 et plus: un polynome peut etre reductible sans avoir de racine, par exemple produit de deux quadratiques.

### Irreductibles dans C[X] et R[X]

Dans `C[X]`:

```text
Irreductibles = polynomes de degre 1.
```

Dans `R[X]`:

```text
Irreductibles = degre 1
ou degre 2 avec discriminant strictement negatif.
```

Donc tout polynome de degre `>= 3` est reductible dans `R[X]`.

### Irreductibles dans F_p[X]

Pour `P in F_p[X]`:

- Degre 1: toujours irreductible.
- Degre 2 ou 3: irreductible si et seulement si aucune racine dans `F_p`.
- Degre 4: aucune racine ne suffit pas; il faut aussi exclure les facteurs de degre 2.

Exemple dans `F_2[X]`:

```text
Le seul irreductible de degre 2 est X^2 + X + 1.
```

Pour montrer qu'un quartique `P` est irreductible dans `F_2[X]`:

1. Verifier `P(0) != 0` et `P(1) != 0`.
2. Verifier que `X^2+X+1` ne divise pas `P`.

Pour `F_3[X]`, les polynomes irreductibles unitaires de degre 2 sont ceux sans racine dans `{0,1,2}`.

### Critere d'Eisenstein

Soit:

```text
P(X) = a_n X^n + ... + a_1 X + a_0 in Z[X].
```

S'il existe un premier `p` tel que:

```text
p divise a_0,a_1,...,a_{n-1}
p ne divise pas a_n
p^2 ne divise pas a_0
```

alors `P` est irreductible dans `Q[X]`.

Si `P` est primitif, alors il est irreductible dans `Z[X]`. Tout polynome unitaire est primitif.

Exemple type:

```text
X^67 + 67X^42 + 268X^9 + 603X^3 + 4623
```

Tester Eisenstein avec `p = 67`:

- `67` divise tous les coefficients sauf le dominant.
- `67` ne divise pas `1`.
- verifier si `67^2` ne divise pas le terme constant.

### Translation et Eisenstein

L'irreductibilite est invariante par translation:

```text
P(X) irreductible <=> P(X+a) irreductible.
```

Reflexe:

```text
Si Eisenstein ne marche pas sur P(X), tester P(X+1) ou P(X-1).
```

Exemples du cours et du CC:

```text
X^4 + X^3 + X^2 + X + 1
```

On calcule:

```text
P(X+1) = X^4 + 5X^3 + 10X^2 + 10X + 5
```

Eisenstein avec `p=5`.

Autre exemple:

```text
X^4 + 1
```

On calcule:

```text
P(X+1) = X^4 + 4X^3 + 6X^2 + 4X + 2
```

Eisenstein avec `p=2`, donc irreductible sur `Q[X]` et sur `Z[X]` car unitaire.

### Reduction modulo p

Soit `P in Z[X]` un polynome unitaire. Si sa reduction modulo `p`, notee `P_bar`, est irreductible dans `F_p[X]`, alors:

```text
P est irreductible dans Z[X].
```

Preuve type par contraposee:

```text
Supposons P reductible dans Z[X].
Alors P = QR avec deg Q, deg R >= 1.
Comme P est unitaire, on peut prendre Q et R unitaires.
En reduisant modulo p, les degres de Q et R ne baissent pas.
Donc P_bar = Q_bar R_bar est reductible dans F_p[X].
```

La reciproque est fausse:

```text
P irreductible dans Z[X] n'implique pas P_bar irreductible dans F_p[X].
```

Exemple du CC:

```text
X^4 + 1 est irreductible dans Z[X],
mais modulo 2: X^4+1 = (X^2+1)^2.
```

---

## 4. Corps finis: minimum operationnel

Meme si on saute la theorie generale des anneaux quotients, il faut savoir utiliser les corps finis construits avec des polynomes irreductibles.

### Construction

Soit `p` premier et `P in F_p[X]` irreductible de degre `n`.

Alors:

```text
K = F_p[X] / <P(X)>
```

est un corps fini de cardinal:

```text
|K| = p^n.
```

Ce qu'il faut retenir pour calculer:

- Tout element de `K` se represente par un polynome de degre `< n`.
- Si `alpha` designe la classe de `X`, alors:

```text
P(alpha) = 0.
```

Cette relation sert a reduire les puissances de `alpha`.

### Exemple fondamental

Dans:

```text
K = F_2[X] / <X^2+X+1>
```

on pose `alpha = X`. Alors:

```text
alpha^2 + alpha + 1 = 0
```

Dans `F_2`, `-1 = 1`, donc:

```text
alpha^2 = alpha + 1.
```

Les elements sont:

```text
0, 1, alpha, alpha+1.
```

### Exemple du CC

Dans:

```text
K = F_2[X] / <P(X)>,  P(X)=X^4+X+1
```

`P` est irreductible, donc `K` est un corps de cardinal:

```text
2^4 = 16.
```

En posant `alpha = X`:

```text
alpha^4 + alpha + 1 = 0
```

donc:

```text
alpha^4 = alpha + 1.
```

Pour trouver l'inverse de `X^5`, deux methodes:

Methode puissance:

```text
K^* a 15 elements, donc alpha^15 = 1.
Donc (alpha^5)^{-1} = alpha^{10}.
```

Puis reduire `alpha^10` avec `alpha^4 = alpha+1`.

Methode Bezout:

```text
X^5 = X(X^4) = X(X+1) = X^2+X dans K.
```

On cherche l'inverse de `X^2+X` modulo `P`.  
Dans le corrige:

```text
P = (X^2+X)(X^2+X+1) + 1.
```

Donc dans `K`:

```text
(X^2+X)(X^2+X+1) = 1.
```

Ainsi:

```text
(X^5)^{-1} = X^2 + X + 1.
```

---

## 5. Methodes d'examen par type de question

### Montrer qu'un ensemble est un groupe

Verifier:

1. Stabilite.
2. Associativite.
3. Element neutre.
4. Inverse de chaque element.

Si c'est une partie d'un groupe connu, utiliser plutot le critere de sous-groupe.

### Montrer qu'une partie est un sous-groupe

Utiliser directement:

```text
H non vide
pour tous x,y in H, xy^{-1} in H
```

En additif:

```text
x-y in H.
```

### Montrer qu'un ensemble est un anneau ou sous-anneau

Si c'est un sous-ensemble d'un anneau connu, utiliser:

```text
B non vide
x-y in B
xy in B
```

Puis dire quelles proprietes viennent de l'anneau ambiant.

### Montrer qu'un ensemble est un corps

Deux strategies:

1. Par definition: montrer que c'est un anneau unitaire non nul et que tout element non nul est inversible.
2. Pour `Z/nZ`: utiliser le theoreme `Z/nZ` corps `<=> n` premier.
3. Pour `F_p[X]/<P>`: montrer que `P` est irreductible, puis conclure que c'est un corps de cardinal `p^{deg P}`.

### Calculer un PGCD de polynomes

Template:

```text
On applique l'algorithme d'Euclide.
A = BQ_1 + R_2
B = R_2Q_2 + R_3
...
Le dernier reste non nul est R_k.
Donc PGCD(A,B) = normalise unitaire de R_k.
```

Pendant le calcul, tu peux multiplier/diviser un reste par une constante non nulle pour simplifier.

### Trouver une relation de Bezout

Faire Euclide puis remonter:

```text
R_k = R_{k-2} - Q_{k-1}R_{k-1}
```

Remplacer progressivement chaque reste par son expression en fonction de `A` et `B`.

Objectif final:

```text
AU + BV = PGCD(A,B).
```

### Tester l'irreductibilite dans Q[X] ou Z[X]

Checklist:

1. Degre 1: irreductible.
2. Degre 2 ou 3: chercher les racines rationnelles possibles.
3. Tester Eisenstein.
4. Si Eisenstein echoue, tester une translation `P(X+1)` ou `P(X-1)`.
5. Tester une reduction modulo un petit premier `p`.

Attention:

- Pour utiliser reduction modulo `p` proprement dans le cours, `P` doit etre unitaire.
- Si `P_bar` est irreductible dans `F_p[X]`, alors `P` est irreductible dans `Z[X]`.

### Tester l'irreductibilite dans F_p[X]

Degre 2 ou 3:

```text
Tester toutes les valeurs de F_p.
Aucune racine <=> irreductible.
```

Degre 4:

```text
1. Tester les racines.
2. Tester les facteurs de degre 2 irreductibles.
```

Dans `F_2[X]`, pour un quartique:

```text
Tester 0, 1, puis diviser par X^2+X+1.
```

Dans `F_3[X]`, enumerer les quadratiques irreductibles unitaires ou verifier par division selon le cas.

### Montrer qu'un quotient polynomial est un corps fini

Template:

```text
On travaille dans F_p[X].
Le polynome P est irreductible de degre n.
Donc K = F_p[X]/<P> est un corps fini.
Son cardinal est p^n.
```

Puis donner la relation:

```text
Si alpha = X, alors P(alpha)=0.
```

### Calculer un inverse dans un corps fini

Methode 1: reduction de puissances.

```text
Utiliser P(alpha)=0 pour reduire les puissances.
```

Methode 2: Bezout.

Pour inverser `A` dans `F_p[X]/<P>`:

```text
Trouver U,V tels que AU + PV = 1.
Alors A^{-1} = U dans le corps fini.
```

Methode 3: ordre du groupe multiplicatif.

Si `K` a `q` elements:

```text
K^* a q-1 elements.
Pour a != 0, a^{q-1}=1.
Donc a^{-1}=a^{q-2}.
```

---

## 6. Pieges frequents

### Piege 1: "Pas de racine" ne veut pas toujours dire irreductible

Vrai seulement pour les degres 2 et 3 sur un corps.

Faux en degre 4:

```text
X^4 - 5X^2 + 6 = (X^2-2)(X^2-3)
```

Pas de racine rationnelle, mais reductible dans `Q[X]`.

### Piege 2: Dans R[X], tout degre >= 3 est reductible

Donc un quartique peut etre irreductible sur `Q[X]` mais pas sur `R[X]`.

Exemple du TD:

```text
X^4 + 30X^2 + 20
```

- Pas irreductible sur `R[X]` car degre 4.
- Irreductible sur `Q[X]` par Eisenstein avec `p=5`.

### Piege 3: Reduction modulo p: la reciproque est fausse

Si `P_bar` irreductible modulo `p`, alors `P` irreductible sur `Z[X]`.

Mais si `P` est irreductible sur `Z[X]`, il peut devenir reductible modulo `p`.

### Piege 4: Dans F_p, toujours reduire les coefficients

Exemples:

```text
Dans F_2: -1 = 1, 2 = 0.
Dans F_3: -1 = 2.
Dans F_5: 6 = 1, -2 = 3.
```

### Piege 5: Ne pas oublier de rendre le PGCD unitaire

Dans `Q[X]`, si le dernier reste est:

```text
2X - 6
```

alors:

```text
PGCD = X - 3.
```

---

## 7. Ce qui est tombe / tres probable

Le CC donne une bonne indication des priorites:

1. Enoncer pourquoi `K[X]` est un anneau commutatif unitaire integre.
2. Dire pourquoi `K[X]` n'est pas un corps.
3. Calculer un PGCD avec Euclide.
4. Justifier qu'un polynome est reductible par factorisation evidente.
5. Prouver une irreductibilite par translation + Eisenstein.
6. Prouver le lemme de reduction modulo `p` par contraposee.
7. Utiliser ce lemme pour montrer une irreductibilite dans `Z[X]`.
8. Montrer que `X^2+X+1` est le seul irreductible de degre 2 sur `F_2[X]`.
9. Montrer qu'un quartique sur `F_2` est irreductible.
10. Construire un corps fini `F_2[X]/<P>`, donner son cardinal.
11. Calculer un inverse dans ce corps fini.

---

## 8. Mini-fiches de redaction

### Fiche: K[X] n'est pas un corps

```text
Le polynome X n'est pas inversible.
En effet, s'il existait Q in K[X] tel que XQ=1,
alors deg(XQ)=1+deg(Q) >= 1, alors que deg(1)=0.
Contradiction.
Donc K[X] n'est pas un corps.
```

### Fiche: irreductibilite d'un cubique dans F_p[X]

```text
Comme deg(P)=3, P est irreductible sur F_p[X]
si et seulement si P n'a aucune racine dans F_p.
On teste les p valeurs.
...
Aucune valeur n'annule P, donc P est irreductible.
```

### Fiche: quartique dans F_2[X]

```text
On teste d'abord les racines:
P(0)=..., P(1)=...
Donc P n'a pas de facteur de degre 1.

Le seul polynome irreductible de degre 2 sur F_2[X] est X^2+X+1.
On effectue la division euclidienne de P par X^2+X+1.
Le reste est non nul, donc X^2+X+1 ne divise pas P.

Ainsi P n'a aucun facteur de degre 1 ou 2.
Comme deg(P)=4, P est irreductible.
```

### Fiche: Eisenstein apres translation

```text
On calcule P(X+1).
On constate que tous les coefficients sauf le dominant sont divisibles par p,
que p ne divise pas le coefficient dominant,
et que p^2 ne divise pas le terme constant.
Par Eisenstein, P(X+1) est irreductible sur Q[X].
Comme l'irreductibilite est invariante par translation,
P(X) est irreductible sur Q[X].
Comme P est unitaire, il est primitif, donc irreductible sur Z[X].
```

### Fiche: reduction modulo p

```text
Supposons par contraposee que P soit reductible dans Z[X].
Alors P=QR avec deg Q, deg R >= 1.
Comme P est unitaire, on peut prendre Q et R unitaires.
En reduisant modulo p, les degres ne baissent pas.
Donc P_bar=Q_bar R_bar est reductible dans F_p[X].
Ainsi, si P_bar est irreductible dans F_p[X], alors P est irreductible dans Z[X].
```

### Fiche: corps fini et cardinal

```text
Le polynome P est irreductible dans F_p[X] et deg(P)=n.
Donc K=F_p[X]/<P> est un corps fini.
Son cardinal est p^n.
En posant alpha=X, on a P(alpha)=0 dans K.
```

### Fiche: inverse dans un corps fini par Bezout

```text
Pour inverser A dans K=F_p[X]/<P>, on cherche U,V tels que:
AU + PV = 1.
Alors dans K, comme P=0, on obtient AU=1.
Donc A^{-1}=U.
```

---

## 9. Derniere checklist avant examen

Savoir faire sans hesitation:

- Donner les definitions: groupe, sous-groupe, anneau, sous-anneau, ideal, corps.
- Utiliser les criteres: sous-groupe, sous-anneau, ideal.
- Expliquer pourquoi `Z/nZ` est un corps si et seulement si `n` est premier.
- Enoncer et utiliser la division euclidienne dans `K[X]`.
- Calculer un PGCD de polynomes avec Euclide.
- Remonter Euclide pour Bezout.
- Tester des racines rationnelles.
- Tester l'irreductibilite dans `F_p[X]`.
- Utiliser Eisenstein, avec ou sans translation.
- Utiliser la reduction modulo `p`.
- Construire `F_p[X]/<P>` quand `P` est irreductible.
- Reduire les puissances avec la relation `P(alpha)=0`.
- Calculer un inverse dans un corps fini.

