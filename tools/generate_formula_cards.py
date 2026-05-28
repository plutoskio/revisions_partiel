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


def draw_formula(
    path: str,
    title: str,
    lines: list[str],
    note: str,
    palette_name: str,
) -> None:
    palette = PALETTE[palette_name]
    width = 1120
    line_count = len(lines)
    height = 130 + 30 * (line_count - 1) + (28 if note else 0)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(' '.join(lines))}</desc>",
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="{palette["bg"]}"/>',
        f'<rect x="14" y="14" width="{width - 28}" height="{height - 28}" rx="11" fill="#ffffff" stroke="{palette["border"]}" stroke-width="2"/>',
        f'<rect x="34" y="32" width="7" height="{height - 64}" rx="4" fill="{palette["accent"]}"/>',
        f'<text x="58" y="55" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="17" font-weight="700" fill="{palette["label"]}">{escape(title)}</text>',
    ]
    y = 86
    for line in lines:
        parts.append(
            f'<text x="58" y="{y}" font-family="SFMono-Regular, Menlo, Consolas, monospace" '
            f'font-size="24" fill="{palette["formula"]}">{escape(line)}</text>'
        )
        y += 30
    if note:
        parts.append(f'<text x="58" y="{height - 30}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="15" fill="{palette["note"]}">{escape(note)}</text>')
    parts.append("</svg>")

    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts) + "\n", encoding="utf-8")


def inline_supervised_formulas() -> None:
    en = "data/supervised_learning/formulas_en"
    fr = "data/supervised_learning/formulas_fr"
    entries = [
        ("inline_01_target_probability.svg", "Predicted probability", "Probabilité prédite", ["P(y = 1 | X)"], "Output of a binary classifier.", "Sortie d'un classifieur binaire."),
        ("inline_02_linear_score.svg", "Linear score", "Score linéaire", ["z = β₀ + β₁x₁ + ... + βₚxₚ"], "Unbounded score before the sigmoid.", "Score non borné avant la sigmoïde."),
        ("inline_03_sigmoid.svg", "Sigmoid function", "Fonction sigmoïde", ["σ(z) = 1 / (1 + exp(-z))"], "Maps a real score to [0, 1].", "Transforme un score réel en valeur entre 0 et 1."),
        ("inline_04_logistic_probability.svg", "Logistic probability", "Probabilité logistique", ["P(y = 1 | X) = σ(β₀ + β₁x₁ + ... + βₚxₚ)"], "Linear model plus sigmoid.", "Modèle linéaire puis sigmoïde."),
        ("inline_05_odds.svg", "Odds", "Odds", ["odds = p / (1 - p)"], "Ratio between positive probability and negative probability.", "Rapport entre probabilité positive et probabilité négative."),
        ("inline_06_log_odds.svg", "Log-odds", "Log-odds", ["log(odds) = log(p / (1 - p))"], "Logarithm of the odds.", "Logarithme des odds."),
        ("inline_07_logit_model.svg", "Logit model", "Modèle logit", ["log(p / (1 - p)) = β₀ + β₁x₁ + ... + βₚxₚ"], "Coefficients are linear on log-odds.", "Les coefficients sont linéaires sur les log-odds."),
        ("inline_08_threshold.svg", "Default threshold rule", "Règle de seuil par défaut", ["predict positive if P(y = 1 | X) > 0.5"], "The threshold can be changed for business costs.", "Le seuil peut être changé selon les coûts métier."),
        ("inline_09_decision_boundary.svg", "Decision boundary", "Frontière de décision", ["β₀ + β₁x₁ + ... + βₚxₚ = 0"], "Because σ(0) = 0.5.", "Parce que σ(0) = 0.5."),
        ("inline_10_coefficient_interpretation.svg", "Coefficient interpretation", "Interprétation d'un coefficient", ["log-odds change by βⱼ", "odds are multiplied by exp(βⱼ)"], "Holding the other variables constant.", "En gardant les autres variables constantes."),
        ("inline_11_binary_target.svg", "Binary target", "Cible binaire", ["y ∈ {0, 1}"], "Bernoulli outcome.", "Variable de Bernoulli."),
        ("inline_12_binary_cross_entropy.svg", "Binary cross-entropy / log-loss", "Binary cross-entropy / log-loss", ["J(β) = -(1/n) Σᵢ [ yᵢ log(ŷᵢ)", "      + (1 - yᵢ) log(1 - ŷᵢ) ]"], "Strongly penalizes confident wrong predictions.", "Pénalise fortement les prédictions confiantes mais fausses."),
        ("inline_13_gradient_update.svg", "Gradient descent update", "Mise à jour par gradient", ["β_new = β_old - η · ∇J(β)"], "η is the learning rate.", "η est le learning rate."),
        ("inline_14_logistic_gradient.svg", "Gradient intuition", "Intuition du gradient", ["Σᵢ (ŷᵢ - yᵢ) · xᵢⱼ"], "Prediction error times feature value.", "Erreur de prédiction multipliée par la variable."),
        ("inline_15_standardization.svg", "Standardization", "Standardisation", ["x_scaled = (x - mean) / standard_deviation"], "Needed for stable gradient-based training.", "Utile pour stabiliser l'optimisation par gradient."),
        ("inline_16_softmax.svg", "Softmax probability", "Probabilité softmax", ["P(y = k | X) = exp(zₖ) / Σⱼ exp(zⱼ)"], "Multi-class probabilities sum to 1.", "Les probabilités multi-classes somment à 1."),
        ("inline_17_accuracy.svg", "Accuracy", "Accuracy", ["Accuracy = (TP + TN) / (TP + TN + FP + FN)"], "Correct predictions over all examples.", "Prédictions correctes sur tous les exemples."),
        ("inline_18_precision.svg", "Precision", "Precision", ["Precision = TP / (TP + FP)"], "Trustworthiness of positive predictions.", "Fiabilité des prédictions positives."),
        ("inline_19_recall.svg", "Recall", "Recall", ["Recall = TP / (TP + FN)"], "How many real positives were found.", "Combien de vrais positifs ont été trouvés."),
        ("inline_20_specificity.svg", "Specificity", "Specificity", ["Specificity = TN / (TN + FP)"], "How many real negatives were rejected.", "Combien de vrais négatifs ont été rejetés."),
        ("inline_21_fpr.svg", "False positive rate", "False positive rate", ["FPR = 1 - specificity = FP / (FP + TN)"], "X-axis of the ROC curve.", "Axe X de la courbe ROC."),
        ("inline_22_f1.svg", "F1 score", "F1-score", ["F1 = 2 · precision · recall / (precision + recall)"], "Harmonic mean of precision and recall.", "Moyenne harmonique de precision et recall."),
        ("inline_23_fbeta.svg", "F-beta score", "F-beta score", ["F_β = (1 + β²) · precision · recall", "      / (β² · precision + recall)"], "β > 1 favors recall; β < 1 favors precision.", "β > 1 favorise recall ; β < 1 favorise precision."),
        ("inline_24_cost.svg", "Cost-based threshold", "Seuil par coût", ["Total cost = FP · Cost(FP) + FN · Cost(FN)"], "Choose the threshold that minimizes this.", "Choisir le seuil qui minimise ce coût."),
        ("inline_25_auc.svg", "AUC interpretation", "Interprétation de l'AUC", ["AUC = P(score_positive > score_negative)"], "Ranking quality, not calibration.", "Qualité de ranking, pas calibration."),
        ("inline_26_gini.svg", "Gini impurity", "Impureté de Gini", ["Gini(D) = 1 - Σᵢ pᵢ²"], "0 means pure.", "0 signifie pur."),
        ("inline_27_weighted_gini.svg", "Weighted Gini split", "Gini pondéré du split", ["Gini_split = (|D_left|/|D|)Gini(D_left)", "           + (|D_right|/|D|)Gini(D_right)"], "CART minimizes weighted child impurity.", "CART minimise l'impureté pondérée des enfants."),
        ("inline_28_entropy.svg", "Entropy", "Entropie", ["H(D) = -Σᵢ pᵢ log₂(pᵢ)"], "Alternative impurity measure.", "Autre mesure d'impureté."),
        ("inline_29_information_gain.svg", "Information gain", "Information gain", ["IG(D, split) = H(D) - weighted_child_entropy"], "Decrease in entropy after splitting.", "Baisse d'entropie après le split."),
        ("inline_30_tree_mse.svg", "Regression tree MSE", "MSE d'un arbre de régression", ["MSE(t) = (1/|D_t|) Σ_{i∈D_t}(yᵢ - ȳ_t)²"], "Regression trees minimize squared error.", "Les arbres de régression minimisent l'erreur quadratique."),
        ("inline_31_ccp.svg", "Cost-complexity pruning", "Cost-complexity pruning", ["R_α(T) = R(T) + α|T|"], "α penalizes tree size.", "α pénalise la taille de l'arbre."),
        ("inline_32_importance.svg", "Tree feature importance", "Importance des variables", ["Importance(feature) = Σ weighted impurity decreases"], "Computed over splits using that feature.", "Calculée sur les splits utilisant cette variable."),
        ("inline_33_forest_vote.svg", "Random forest classification", "Classification par forêt aléatoire", ["ŷ = majority_vote(T₁(x), ..., T_B(x))"], "One vote per tree.", "Un vote par arbre."),
        ("inline_34_forest_average.svg", "Random forest regression", "Régression par forêt aléatoire", ["ŷ = (1/B) Σ_b T_b(x)"], "Average of tree predictions.", "Moyenne des prédictions des arbres."),
        ("inline_35_variance_mean.svg", "Variance of an average", "Variance d'une moyenne", ["Var(mean) = σ² / B"], "Only true for independent models.", "Vrai pour des modèles indépendants."),
        ("inline_36_variance_forest.svg", "Variance with correlated trees", "Variance avec arbres corrélés", ["Var(forest) = ρσ² + ((1 - ρ)/B)σ²"], "Random features lower ρ.", "Les variables aléatoires diminuent ρ."),
        ("inline_37_probability_definition.svg", "Probability shorthand", "Notation de probabilité", ["p = P(y = 1)"], "p is the predicted probability of the positive class.", "p est la probabilité prédite de la classe positive."),
        ("inline_38_threshold_variable.svg", "Threshold decision rule", "Règle de décision avec seuil", ["predict positive if P(y = 1 | X) > threshold"], "Raising the threshold usually raises precision and lowers recall.", "Augmenter le seuil augmente souvent precision et baisse recall."),
        ("inline_39_pr_perfect.svg", "Perfect precision-recall point", "Point precision-recall parfait", ["recall = 1  and  precision = 1"], "The ideal top-right point of a PR curve.", "Le point idéal en haut à droite d'une courbe PR."),
        ("inline_40_pr_baseline.svg", "PR baseline", "Baseline de courbe PR", ["positive class rate = positives / total examples"], "Random ranking starts around the positive class rate.", "Un ranking aléatoire commence autour du taux de classe positive."),
        ("inline_41_leaf_prediction.svg", "Regression tree leaf prediction", "Prédiction d'une feuille de régression", ["leaf prediction = mean target value in the leaf"], "Trees predict an average inside each leaf.", "L'arbre prédit la moyenne dans chaque feuille."),
        ("inline_42_axis_splits.svg", "Axis-aligned tree splits", "Splits alignés sur les axes", ["x₁ ≤ threshold", "x₂ ≤ threshold"], "Decision trees split one feature at a time.", "Les arbres coupent une variable à la fois."),
    ]
    for filename, title_en, title_fr, lines, note_en, note_fr in entries:
        draw_formula(f"{en}/{filename}", title_en, lines, note_en, "supervised")
        draw_formula(f"{fr}/{filename}", title_fr, lines, note_fr, "supervised")


def inline_unsupervised_formulas() -> None:
    folder = "data/unsupervised_learning/formulas"
    entries = [
        ("inline_01_hard_clustering.svg", "Hard clustering", ["P(Cₖ | xᵢ) ∈ {0, 1}"], "Un point appartient à un seul cluster."),
        ("inline_02_soft_clustering.svg", "Soft clustering", ["P(Cₖ | xᵢ) ∈ [0, 1]"], "Un point reçoit des probabilités d'appartenance."),
        ("inline_03_wcss.svg", "WCSS / inertie", ["WCSS = Σ distances²(point, centroïde)"], "Mesure la compacité des clusters."),
        ("inline_04_kmeans_objective.svg", "Objectif K-Means", ["min Σₖ Σ_{xᵢ∈Cₖ} ||xᵢ - μₖ||²"], "Minimise la distance au carré aux centroïdes."),
        ("inline_05_euclidean.svg", "Distance euclidienne", ["d(x,c) = sqrt(Σⱼ (xⱼ - cⱼ)²)"], "Distance utilisée par K-Means."),
        ("inline_06_standardization.svg", "Standardisation", ["x_scaled = (x - mean) / standard_deviation"], "Indispensable pour les méthodes basées sur les distances."),
        ("inline_07_contamination.svg", "Contamination", ["contamination = 0.01"], "On s'attend à environ 1 % d'anomalies."),
        ("inline_08_lof_normal.svg", "LOF normal", ["LOF ≈ 1"], "Densité similaire à celle des voisins."),
        ("inline_09_lof_anomaly.svg", "LOF suspect", ["LOF >> 1"], "Beaucoup moins dense que ses voisins."),
    ]
    for filename, title, lines, note in entries:
        draw_formula(f"{folder}/{filename}", title, lines, note, "unsupervised")


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
    inline_supervised_formulas()
    inline_unsupervised_formulas()


if __name__ == "__main__":
    main()
