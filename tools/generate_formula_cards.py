from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PALETTE = {
    "supervised": {
        "bg": "#f8fbff",
        "border": "#b9d4f5",
        "title": "#17324d",
        "accent": "#286fb4",
        "row": "#ffffff",
        "row_alt": "#eef6ff",
        "label": "#24415f",
        "formula": "#111827",
        "note": "#50657a",
    },
    "unsupervised": {
        "bg": "#fbfaf7",
        "border": "#dbc9a5",
        "title": "#3d321f",
        "accent": "#8a6424",
        "row": "#ffffff",
        "row_alt": "#f7f1e6",
        "label": "#55442b",
        "formula": "#111827",
        "note": "#6f624d",
    },
}


def draw_card(
    path: str,
    title: str,
    subtitle: str,
    rows: list[tuple[str, list[str], str]],
    palette_name: str,
) -> None:
    palette = PALETTE[palette_name]
    width = 1120
    top = 34
    row_gap = 10
    row_heights = [62 + 24 * (len(lines) - 1) + (18 if note else 0) for _, lines, note in rows]
    height = top + 82 + sum(row_heights) + row_gap * (len(rows) - 1) + 34

    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(subtitle)}</desc>",
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="{palette["bg"]}"/>',
        f'<rect x="16" y="16" width="{width - 32}" height="{height - 32}" rx="14" fill="none" stroke="{palette["border"]}" stroke-width="2"/>',
        f'<rect x="36" y="34" width="8" height="50" rx="4" fill="{palette["accent"]}"/>',
        f'<text x="60" y="58" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="{palette["title"]}">{escape(title)}</text>',
        f'<text x="60" y="82" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="16" fill="{palette["note"]}">{escape(subtitle)}</text>',
    ]

    y = 108
    for i, (label, lines, note) in enumerate(rows):
        h = row_heights[i]
        fill = palette["row"] if i % 2 == 0 else palette["row_alt"]
        parts.append(f'<rect x="36" y="{y}" width="{width - 72}" height="{h}" rx="10" fill="{fill}" stroke="{palette["border"]}" stroke-width="1"/>')
        parts.append(f'<text x="58" y="{y + 31}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="17" font-weight="700" fill="{palette["label"]}">{escape(label)}</text>')

        formula_y = y + 31
        for j, line in enumerate(lines):
            parts.append(
                f'<text x="360" y="{formula_y + 24 * j}" font-family="SFMono-Regular, Menlo, Consolas, monospace" '
                f'font-size="20" fill="{palette["formula"]}">{escape(line)}</text>'
            )
        if note:
            parts.append(f'<text x="360" y="{y + h - 18}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="15" fill="{palette["note"]}">{escape(note)}</text>')
        y += h + row_gap

    parts.append("</svg>")

    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts) + "\n", encoding="utf-8")


def supervised_cards() -> None:
    logistic_rows_en = [
        ("Linear score", ["z = β₀ + β₁x₁ + ... + βₚxₚ"], "Unbounded score before converting to a probability."),
        ("Sigmoid", ["σ(z) = 1 / (1 + exp(-z))"], "Squashes any real score into the interval [0, 1]."),
        ("Probability", ["P(y = 1 | X) = σ(β₀ + β₁x₁ + ... + βₚxₚ)"], "The model predicts a probability, then a threshold creates a class label."),
        ("Log-odds", ["log(p / (1 - p)) = β₀ + β₁x₁ + ... + βₚxₚ"], "Coefficients are linear on log-odds, not directly on probability."),
        ("Cross-entropy", ["J(β) = -(1/n) Σᵢ [yᵢ log(ŷᵢ)", "      + (1 - yᵢ) log(1 - ŷᵢ)]"], "Confident wrong predictions receive a large penalty."),
        ("Gradient step", ["β_new = β_old - η · ∇J(β)"], "η is the learning rate."),
        ("Softmax", ["P(y = k | X) = exp(zₖ) / Σⱼ exp(zⱼ)"], "Multi-class extension: all class probabilities sum to 1."),
    ]
    logistic_rows_fr = [
        ("Score linéaire", ["z = β₀ + β₁x₁ + ... + βₚxₚ"], "Score non borné avant transformation en probabilité."),
        ("Sigmoïde", ["σ(z) = 1 / (1 + exp(-z))"], "Transforme n'importe quel score réel en valeur entre 0 et 1."),
        ("Probabilité", ["P(y = 1 | X) = σ(β₀ + β₁x₁ + ... + βₚxₚ)"], "Le modèle prédit une probabilité, puis un seuil crée la classe."),
        ("Log-odds", ["log(p / (1 - p)) = β₀ + β₁x₁ + ... + βₚxₚ"], "Les coefficients sont linéaires sur les log-odds, pas sur la probabilité."),
        ("Cross-entropy", ["J(β) = -(1/n) Σᵢ [yᵢ log(ŷᵢ)", "      + (1 - yᵢ) log(1 - ŷᵢ)]"], "Une prédiction confiante mais fausse est fortement pénalisée."),
        ("Descente de gradient", ["β_new = β_old - η · ∇J(β)"], "η est le learning rate."),
        ("Softmax", ["P(y = k | X) = exp(zₖ) / Σⱼ exp(zⱼ)"], "Extension multi-classe : les probabilités somment à 1."),
    ]

    metrics_rows_en = [
        ("Accuracy", ["(TP + TN) / (TP + TN + FP + FN)"], "Share of all predictions that are correct."),
        ("Precision", ["TP / (TP + FP)"], "Among predicted positives, how many are truly positive?"),
        ("Recall / TPR", ["TP / (TP + FN)"], "Among real positives, how many did we find?"),
        ("Specificity / FPR", ["Specificity = TN / (TN + FP)", "FPR = FP / (FP + TN) = 1 - specificity"], "Performance on the negative class."),
        ("F1", ["2 · precision · recall / (precision + recall)"], "Harmonic mean; punishes imbalance between precision and recall."),
        ("F-beta", ["(1 + β²) · precision · recall / (β² · precision + recall)"], "β > 1 favors recall; β < 1 favors precision."),
        ("Cost threshold", ["Total cost = FP · Cost(FP) + FN · Cost(FN)"], "Choose the threshold from business cost, not automatically 0.5."),
        ("AUC", ["P(score_positive > score_negative)"], "Ranking quality, not probability calibration."),
    ]
    metrics_rows_fr = [
        ("Accuracy", ["(TP + TN) / (TP + TN + FP + FN)"], "Part de toutes les prédictions qui sont correctes."),
        ("Precision", ["TP / (TP + FP)"], "Parmi les positifs prédits, combien sont vraiment positifs ?"),
        ("Recall / TPR", ["TP / (TP + FN)"], "Parmi les vrais positifs, combien le modèle trouve-t-il ?"),
        ("Specificity / FPR", ["Specificity = TN / (TN + FP)", "FPR = FP / (FP + TN) = 1 - specificity"], "Performance sur la classe négative."),
        ("F1", ["2 · precision · recall / (precision + recall)"], "Moyenne harmonique ; pénalise le déséquilibre precision/recall."),
        ("F-beta", ["(1 + β²) · precision · recall / (β² · precision + recall)"], "β > 1 favorise recall ; β < 1 favorise precision."),
        ("Seuil par coût", ["Coût total = FP · Coût(FP) + FN · Coût(FN)"], "Choisir le seuil selon le coût métier, pas automatiquement 0.5."),
        ("AUC", ["P(score_positif > score_négatif)"], "Qualité de ranking, pas calibration des probabilités."),
    ]

    trees_rows_en = [
        ("Gini impurity", ["Gini(D) = 1 - Σᵢ pᵢ²"], "0 means pure; higher means more mixed."),
        ("Weighted split", ["Gini_split = (|D_left|/|D|)Gini(D_left)", "           + (|D_right|/|D|)Gini(D_right)"], "CART chooses the split with the lowest weighted impurity."),
        ("Entropy", ["H(D) = -Σᵢ pᵢ log₂(pᵢ)"], "Alternative impurity measure."),
        ("Information gain", ["IG(D, split) = H(D) - weighted_child_entropy"], "How much disorder decreases after splitting."),
        ("Regression tree", ["MSE(t) = (1/|D_t|) Σ_{i∈D_t}(yᵢ - ȳ_t)²"], "Leaves predict the mean target value."),
        ("Pruning", ["R_α(T) = R(T) + α|T|"], "α penalizes tree complexity."),
        ("Importance", ["Σ weighted impurity decreases caused by feature"], "Biased toward variables with many possible split points."),
    ]
    trees_rows_fr = [
        ("Impureté de Gini", ["Gini(D) = 1 - Σᵢ pᵢ²"], "0 signifie pur ; plus haut signifie plus mélangé."),
        ("Split pondéré", ["Gini_split = (|D_left|/|D|)Gini(D_left)", "           + (|D_right|/|D|)Gini(D_right)"], "CART choisit le split avec l'impureté pondérée la plus faible."),
        ("Entropie", ["H(D) = -Σᵢ pᵢ log₂(pᵢ)"], "Autre mesure d'impureté."),
        ("Information gain", ["IG(D, split) = H(D) - weighted_child_entropy"], "Mesure la baisse de désordre après un split."),
        ("Arbre de régression", ["MSE(t) = (1/|D_t|) Σ_{i∈D_t}(yᵢ - ȳ_t)²"], "Les feuilles prédisent la moyenne de la cible."),
        ("Élagage", ["R_α(T) = R(T) + α|T|"], "α pénalise la complexité de l'arbre."),
        ("Importance", ["Σ baisses d'impureté pondérées causées par la variable"], "Biaisée vers les variables avec beaucoup de splits possibles."),
    ]

    forest_rows_en = [
        ("Classification", ["ŷ = majority_vote(T₁(x), T₂(x), ..., T_B(x))"], "Each tree votes; the forest predicts the majority class."),
        ("Regression", ["ŷ = (1/B) Σ_b T_b(x)"], "Average the tree predictions."),
        ("Independent trees", ["Var(mean) = σ² / B"], "Averaging many independent noisy models reduces variance."),
        ("Correlated trees", ["Var(forest) = ρσ² + ((1 - ρ)/B)σ²"], "If trees are correlated, variance cannot go below the ρσ² term."),
        ("Out-of-bag", ["P(row not sampled by one tree) ≈ e⁻¹ ≈ 36.8%"], "Rows left out of a bootstrap sample can validate that tree."),
        ("Random features", ["lower feature overlap ⇒ lower ρ"], "The main trick that decorrelates trees."),
    ]
    forest_rows_fr = [
        ("Classification", ["ŷ = vote_majoritaire(T₁(x), T₂(x), ..., T_B(x))"], "Chaque arbre vote ; la forêt prédit la classe majoritaire."),
        ("Régression", ["ŷ = (1/B) Σ_b T_b(x)"], "On moyenne les prédictions des arbres."),
        ("Arbres indépendants", ["Var(moyenne) = σ² / B"], "Moyenner beaucoup de modèles indépendants réduit la variance."),
        ("Arbres corrélés", ["Var(forest) = ρσ² + ((1 - ρ)/B)σ²"], "Si les arbres sont corrélés, la variance ne descend pas sous ρσ²."),
        ("Out-of-bag", ["P(ligne non tirée par un arbre) ≈ e⁻¹ ≈ 36.8%"], "Les lignes hors bootstrap peuvent valider cet arbre."),
        ("Variables aléatoires", ["moins de variables communes ⇒ ρ plus faible"], "Le mécanisme clé qui décorrèle les arbres."),
    ]

    for folder, lang, rows in [
        ("data/supervised_learning/formulas_en", "en", logistic_rows_en),
        ("data/supervised_learning/formulas_fr", "fr", logistic_rows_fr),
    ]:
        draw_card(
            f"{folder}/01_logistic_formulas.svg",
            "Logistic Regression: Formula Map" if lang == "en" else "Régression logistique : carte des formules",
            "From linear score to probability, loss, optimization, and softmax." if lang == "en" else "Du score linéaire à la probabilité, la perte, l'optimisation et softmax.",
            rows,
            "supervised",
        )

    for folder, lang, rows in [
        ("data/supervised_learning/formulas_en", "en", metrics_rows_en),
        ("data/supervised_learning/formulas_fr", "fr", metrics_rows_fr),
    ]:
        draw_card(
            f"{folder}/02_classification_metrics.svg",
            "Classification Metrics: Formula Map" if lang == "en" else "Métriques de classification : carte des formules",
            "Same confusion matrix, different denominators and decision goals." if lang == "en" else "Même matrice de confusion, mais dénominateurs et objectifs différents.",
            rows,
            "supervised",
        )

    for folder, lang, rows in [
        ("data/supervised_learning/formulas_en", "en", trees_rows_en),
        ("data/supervised_learning/formulas_fr", "fr", trees_rows_fr),
    ]:
        draw_card(
            f"{folder}/03_tree_formulas.svg",
            "Decision Trees: Formula Map" if lang == "en" else "Arbres de décision : carte des formules",
            "Impurity, split choice, regression leaves, pruning, and importance." if lang == "en" else "Impureté, choix du split, feuilles de régression, élagage et importance.",
            rows,
            "supervised",
        )

    for folder, lang, rows in [
        ("data/supervised_learning/formulas_en", "en", forest_rows_en),
        ("data/supervised_learning/formulas_fr", "fr", forest_rows_fr),
    ]:
        draw_card(
            f"{folder}/04_random_forest_formulas.svg",
            "Random Forests: Formula Map" if lang == "en" else "Forêts aléatoires : carte des formules",
            "Voting, averaging, out-of-bag intuition, and variance reduction." if lang == "en" else "Vote, moyenne, intuition out-of-bag et réduction de variance.",
            rows,
            "supervised",
        )


def unsupervised_cards() -> None:
    clustering_rows = [
        ("Hard clustering", ["P(Cₖ | xᵢ) ∈ {0, 1}"], "Chaque point appartient à un seul cluster."),
        ("Soft clustering", ["P(Cₖ | xᵢ) ∈ [0, 1]"], "Chaque point reçoit des probabilités d'appartenance."),
        ("Objectif K-Means", ["min Σₖ Σ_{xᵢ∈Cₖ} ||xᵢ - μₖ||²"], "Minimise la distance au carré aux centroïdes."),
        ("WCSS / inertie", ["Σ distances²(point, centroïde du cluster)"], "Diminue toujours quand K augmente."),
        ("Distance euclidienne", ["d(x,c) = sqrt(Σⱼ (xⱼ - cⱼ)²)"], "Très sensible au scaling."),
        ("Standardisation", ["x_scaled = (x - mean) / standard_deviation"], "Met les variables sur une échelle comparable."),
        ("Silhouette", ["s(i) = (b(i) - a(i)) / max(a(i), b(i))"], "a = distance intra-cluster ; b = meilleur autre cluster."),
    ]
    gmm_rows = [
        ("Mélange gaussien", ["p(x) = Σₖ πₖ N(x | μₖ, Σₖ)"], "Le dataset est vu comme une somme de distributions gaussiennes."),
        ("Responsabilité", ["γᵢₖ = πₖN(xᵢ|μₖ,Σₖ) / ΣⱼπⱼN(xᵢ|μⱼ,Σⱼ)"], "Probabilité que le cluster k ait généré le point i."),
        ("Centre", ["μₖ"], "Position moyenne du cluster k."),
        ("Covariance", ["Σₖ"], "Forme, étalement et orientation du cluster."),
        ("Poids", ["πₖ ≥ 0  et  Σₖπₖ = 1"], "Proportion attendue de points dans chaque cluster."),
        ("Mahalanobis", ["d_M(x,μ) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))"], "Distance ajustée par la forme du cluster."),
    ]
    density_rows = [
        ("Single linkage", ["d(A,B) = min d(a,b),  a∈A, b∈B"], "Fusionne selon les deux points les plus proches."),
        ("Complete linkage", ["d(A,B) = max d(a,b),  a∈A, b∈B"], "Fusionne selon les deux points les plus éloignés."),
        ("Average linkage", ["d(A,B) = mean d(a,b),  a∈A, b∈B"], "Compromis plus stable."),
        ("Ward", ["fusion qui augmente le moins la variance intra-cluster"], "Proche de l'idée d'inertie."),
        ("Voisinage DBSCAN", ["N_eps(x) = {xᵢ : d(x, xᵢ) ≤ eps}"], "Tous les points dans le rayon eps."),
        ("Core point", ["|N_eps(x)| ≥ MinPts"], "Un core point peut étendre un cluster."),
        ("Noise", ["ni core, ni border"], "Point traité comme anomalie/bruit."),
    ]
    pca_rows = [
        ("Standardiser", ["X_scaled = (X - μ) / σ"], "Étape presque toujours nécessaire avant PCA."),
        ("Covariance", ["Cov(X_scaled)"], "Résume les variations communes entre variables."),
        ("Axes principaux", ["Cov · W = W · Λ"], "W contient les directions ; Λ contient les variances expliquées."),
        ("Projection", ["Z = X_scaled · W"], "Nouvelles coordonnées sur les composantes principales."),
        ("Variance expliquée", ["ratio_k = λₖ / Σⱼ λⱼ"], "Part d'information capturée par PC k."),
        ("Variance cumulée", ["cumulative_K = Σ_{k=1..K} ratio_k"], "Aide à choisir combien de composantes garder."),
    ]
    anomaly_rows = [
        ("Contamination", ["contamination = proportion attendue d'anomalies"], "Influence le seuil de décision."),
        ("One-Class SVM", ["f(x) ≥ 0 ⇒ normal", "f(x) < 0 ⇒ anomalie"], "Apprend une frontière autour des données normales."),
        ("Paramètre ν", ["ν ≈ borne sur la fraction d'anomalies autorisées"], "Contrôle aussi la souplesse de la frontière."),
        ("Isolation Forest", ["E[h(x)] court ⇒ anomalie"], "Un point rare et différent s'isole en peu de splits."),
        ("LOF", ["LOFₖ(x) = densité_locale(voisins) / densité_locale(x)"], "Compare la densité du point à celle de ses voisins."),
        ("Interprétation LOF", ["LOF ≈ 1 ⇒ normal", "LOF >> 1 ⇒ suspect"], "Utile pour anomalies locales/contextuelles."),
    ]

    folder = "data/unsupervised_learning/formulas"
    draw_card(
        f"{folder}/01_clustering_formulas.svg",
        "Clustering : carte des formules",
        "Assignments, distances, inertie, scaling et silhouette.",
        clustering_rows,
        "unsupervised",
    )
    draw_card(
        f"{folder}/02_gmm_formulas.svg",
        "Gaussian Mixture Models : carte des formules",
        "Soft clustering probabiliste avec moyennes, covariances et poids.",
        gmm_rows,
        "unsupervised",
    )
    draw_card(
        f"{folder}/03_hierarchical_dbscan_formulas.svg",
        "Hiérarchique + DBSCAN : carte des formules",
        "Distances entre clusters et définition d'une zone dense.",
        density_rows,
        "unsupervised",
    )
    draw_card(
        f"{folder}/04_pca_formulas.svg",
        "PCA : carte des formules",
        "Standardisation, projection et variance expliquée.",
        pca_rows,
        "unsupervised",
    )
    draw_card(
        f"{folder}/05_anomaly_formulas.svg",
        "Anomalies : carte des formules",
        "Seuils, frontière normale, isolation et densité locale.",
        anomaly_rows,
        "unsupervised",
    )


def main() -> None:
    supervised_cards()
    unsupervised_cards()


if __name__ == "__main__":
    main()
