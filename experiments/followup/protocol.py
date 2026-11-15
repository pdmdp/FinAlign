"""Fixed grids, ranking rules and the three planned statistical comparisons."""

import math

import numpy as np

from finalign.core import LEXICAL_DEPTH, fuse_passages, metrics, min_max, ranked

WEIGHTS = ((.70, .15, .15), (.50, .25, .25), (.85, .075, .075),
           (.70, .20, .10), (.70, .10, .20))
ALPHAS = (0., .25, .50, .75, 1.)
VARIANTS = ("original", "keywords", "hyde")
SYSTEMS = ("L", "B", "E", "H", "F")
COMPARISONS = (("C1", "E", "B"), ("C2", "H", "E"), ("C3", "F", "H"))
ARCHIVE_SHA256 = "45b9da96081deb1f2f8c946125c413bafe7bad464fcc0b3c9ccec21300036366"
BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED = 20_000, 2026
PERMUTATIONS, PERMUTATION_SEED = 100_000, 2027


def expanded_scores(passages, weights):
    return fuse_passages([(w, passages[name]) for w, name in zip(weights, VARIANTS)])[0]


def blend(dense, lexical, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")
    if alpha == 0:
        return dict(dense)
    if alpha == 1:
        return dict(lexical)
    d = min_max(dict(ranked(dense, LEXICAL_DEPTH)))
    l = min_max(dict(ranked(lexical, LEXICAL_DEPTH)))
    return {doc: (1 - alpha) * d.get(doc, 0.) + alpha * l.get(doc, 0.)
            for doc in sorted(d.keys() | l.keys())}


def effectiveness(scores, qrels):
    result = metrics([doc for doc, _ in ranked(scores)], qrels)
    if result is None:
        raise ValueError("Every evaluation query must have positive judgments.")
    return {k: result[k] for k in ("nDCG@10", "Recall@10")}


def best_candidate(means):
    values = np.asarray(means, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Selection requires finite candidate means.")
    # Candidate order implements both published tie rules.
    tied = np.flatnonzero(np.max(values) - values <= 1e-12).tolist()
    return tied[0], tied


def select_parameters(rows, components):
    if len(rows) != len(components) or not rows:
        raise ValueError("Incomplete development set.")
    expansion = [[effectiveness(expanded_scores(c["passages"], w), r["qrels"])["nDCG@10"]
                  for w in WEIGHTS] for r, c in zip(rows, components)]
    means = np.mean(expansion, axis=0).tolist()
    chosen, ties = best_candidate(means)
    weights = WEIGHTS[chosen]
    lexical = []
    for row, c in zip(rows, components):
        dense = expanded_scores(c["passages"], weights)
        lexical.append([effectiveness(blend(dense, c["lexical"], a), row["qrels"])["nDCG@10"]
                        for a in ALPHAS])
    alpha_means = np.mean(lexical, axis=0).tolist()
    alpha_index, alpha_ties = best_candidate(alpha_means)
    return {"query_ids": [r["qid"] for r in rows], "weights": list(weights),
            "alpha": ALPHAS[alpha_index], "weight_grid": [list(w) for w in WEIGHTS],
            "alpha_grid": list(ALPHAS), "weight_means": means, "alpha_means": alpha_means,
            "weight_ties": ties, "alpha_ties": alpha_ties,
            "weight_choice": chosen, "alpha_choice": alpha_index,
            "per_query_weight_ndcg": expansion, "per_query_alpha_ndcg": lexical}


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values.")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def paired_analysis(differences, *, bootstrap_samples=BOOTSTRAP_SAMPLES,
                    permutations=PERMUTATIONS, batch_size=256):
    delta = np.asarray(differences, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 3 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Expected a complete finite query-by-three difference matrix.")
    if min(bootstrap_samples, permutations, batch_size) < 1:
        raise ValueError("Resampling counts must be positive.")
    n = len(delta)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty((bootstrap_samples, 3))
    for start in range(0, bootstrap_samples, batch_size):
        count = min(batch_size, bootstrap_samples - start)
        indices = rng.integers(0, n, size=(count, n))
        bootstrap[start:start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [.025, .975], axis=0, method="linear")
    rng = np.random.default_rng(PERMUTATION_SEED)
    observed = np.abs(delta.sum(axis=0))
    extreme = np.zeros(3, dtype=np.int64)
    # Roundoff tolerance only; comparisons remain two-sided.
    tolerance = 16 * np.finfo(float).eps * np.maximum(1., np.abs(delta).sum(axis=0))
    for start in range(0, permutations, batch_size):
        count = min(batch_size, permutations - start)
        signs = (rng.integers(0, 2, size=(count, n), dtype=np.int64) * 2 - 1).astype(float)
        extreme += (np.abs(signs @ delta) >= observed - tolerance).sum(axis=0)
    raw = (extreme + 1) / (permutations + 1)
    adjusted = holm(raw)
    output = []
    for i, (name, left, right) in enumerate(COMPARISONS):
        values = delta[:, i]
        output.append({"comparison": name, "difference": f"{left} - {right}", "queries": n,
                       "mean_delta": float(values.mean()), "median_delta": float(np.median(values)),
                       "ci_low": float(low[i]), "ci_high": float(high[i]),
                       "improved": int((values > 0).sum()), "tied": int((values == 0).sum()),
                       "worsened": int((values < 0).sum()), "raw_p": float(raw[i]),
                       "holm_p": float(adjusted[i]), "reject_at_0_05": bool(adjusted[i] <= .05)})
    return output
