"""
==========================================
LinearSVC: multi-class strategies compared
==========================================

:class:`~sklearn.svm.LinearSVC` exposes two multi-class strategies:

* **One-vs-Rest** (``multi_class="ovr"``, the default): one binary classifier
  per class, each separating that class from all others.
* **Crammer–Singer** (``multi_class="crammer_singer"``): a single joint
  optimisation over all classes [#1]_.

A third option wraps :class:`~sklearn.svm.LinearSVC` in a
:class:`~sklearn.multiclass.OneVsOneClassifier`, training one binary
classifier per pair of classes.

We compare all three on two scenarios:

1. **Iris** (2 features, 3 classes) — decision boundaries, qualitative only.
2. **Synthetic, large K** (25 classes, well-separated,
   :func:`~sklearn.datasets.make_classification`) — accuracy and fit time
   in a regime that exposes One-vs-One's quadratic sub-problem overhead.

See :ref:`svm_multi_class` for the user-guide section.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Helpers
# -------
#
# Each strategy is wrapped in ``StandardScaler → LinearSVC``: scaling is
# mandatory because :class:`~sklearn.svm.LinearSVC` is sensitive to feature
# magnitudes.  Shared hyper-parameters: ``tol=1e-3``, ``max_iter=100_000``
# (Crammer-Singer needs the headroom), and ``dual="auto"``.
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris, make_classification
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def make_estimators(random_state=0):
    """Return three named ``StandardScaler → LinearSVC`` pipelines."""
    base = dict(random_state=random_state, dual="auto", tol=1e-3, max_iter=100_000)
    return {
        name: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        for name, clf in [
            ("One-vs-Rest", LinearSVC(multi_class="ovr", **base)),
            ("One-vs-One", OneVsOneClassifier(LinearSVC(**base))),
            ("Crammer-Singer", LinearSVC(multi_class="crammer_singer", **base)),
        ]
    }


def run_cv(X, y, cv):
    """Cross-validate all three strategies on (X, y)."""
    return {
        name: cross_validate(est, X, y, cv=cv, n_jobs=2)
        for name, est in make_estimators().items()
    }


def plot_cv_results(cv_results, dataset_name):
    """Side-by-side: KDE of fold accuracies and bar of mean fit times."""
    scores = pd.DataFrame({n: r["test_score"] for n, r in cv_results.items()})
    fit_times = pd.Series({n: r["fit_time"].mean() for n, r in cv_results.items()})

    fig, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(12, 4))
    scores.plot.kde(ax=ax_acc, legend=True)
    ax_acc.set(xlabel="Accuracy score", title=f"Accuracy — {dataset_name}")
    fit_times.plot.bar(ax=ax_time, rot=15)
    ax_time.set(ylabel="Mean fit time (s)", title=f"Fit time — {dataset_name}")
    fig.suptitle("RepeatedStratifiedKFold, 5 splits x 3 repeats", y=1.02, fontsize=10)
    plt.tight_layout()
    plt.show()


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)

# %%
# Decision boundaries on Iris
# ---------------------------
#
# Iris is restricted to its first two features so decision regions can be
# drawn directly.  Iris is easy enough that all three strategies reach
# near-perfect accuracy — the informative signal here is boundary *shape*.

iris = load_iris()
X_iris, y_iris = iris.data[:, :2], iris.target

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, est) in zip(axes, make_estimators().items()):
    est.fit(X_iris, y_iris)
    DecisionBoundaryDisplay.from_estimator(
        est,
        X_iris,
        ax=ax,
        response_method="predict",
        alpha=0.8,
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
    )
    ax.scatter(*X_iris.T, c=y_iris, cmap="coolwarm", s=20, edgecolors="k")
    ax.set(xticks=(), yticks=(), title=name)
fig.suptitle("Decision boundaries on Iris (first 2 features)", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Reading the boundaries
# ~~~~~~~~~~~~~~~~~~~~~~
#
# All three separate Iris well, but the boundary shapes differ:
#
# * **One-vs-Rest** fits three independent hyperplanes; the most confident
#   binary classifier wins ambiguous central regions.
# * **One-vs-One** fits three pairwise classifiers and resolves ties by
#   majority vote.
# * **Crammer-Singer** solves a joint optimisation over all three classes
#   at once; the resulting boundaries often resemble One-vs-One's but come
#   from a single training pass.
#
# At higher K these structural differences translate into measurable
# accuracy and fit-time gaps — the next cell quantifies them.

# %%
# Quantitative comparison: K = 25, well-separated classes
# -------------------------------------------------------
#
# :func:`~sklearn.datasets.make_classification` with 25 cleanly separated
# classes (``class_sep=2.0``, no label noise) isolates the effect of K
# without confounding from class overlap:
#
# * **One-vs-One** trains ``K * (K - 1) / 2 = 300`` binary sub-problems
#   sequentially inside each fold.
# * **One-vs-Rest** trains ``K = 25`` binary problems on the full training
#   set — linear in K and in ``n_samples``.
# * **Crammer-Singer** solves one joint problem over all 25 classes; its
#   per-iteration cost scales with K but there is only one optimisation.

X_lk, y_lk = make_classification(
    n_samples=1_500,
    n_features=30,
    n_informative=25,
    n_redundant=3,
    n_classes=25,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=0,
)
plot_cv_results(run_cv(X_lk, y_lk, cv), "K=25, well-separated")

# %%
# Conclusion
# ----------
#
# The K = 25 benchmark separates the three strategies cleanly on both
# accuracy and fit time:
#
# * **One-vs-One** — most accurate, by far the slowest.  Each of its 300
#   pairwise sub-problems is easy (only two classes), but
#   :class:`~sklearn.multiclass.OneVsOneClassifier` runs them sequentially
#   inside each fold; the ``n_jobs`` argument to
#   :func:`~sklearn.model_selection.cross_validate` only parallelises across
#   folds, not within.
# * **One-vs-Rest** — fastest by a wide margin.  K = 25 binary problems is
#   ~8 % of One-vs-One's work, and the resulting accuracy sits in the
#   middle of the three.
# * **Crammer-Singer** — intermediate on both axes.  One joint optimisation
#   beats 300 sub-problems but loses to 25.  Its accuracy lags here because
#   the joint objective is harder to optimise as K grows; under heavy class
#   overlap (not shown) the gap widens further.
#
# **Practical guidance**
#
# * **One-vs-Rest** is the safest default — fast, predictable, robust to
#   both large K and class overlap.  This matches the
#   :ref:`svm_multi_class` user-guide recommendation.
# * **Crammer-Singer** can match One-vs-Rest's accuracy when classes are
#   cleanly separated and K is moderate, but degrades under heavy overlap.
#   Profile on the target dataset before preferring it.
# * **One-vs-One** is often the most accurate but costs K² / 2 sub-problems.
#   Use when K is small or when accuracy is the priority.
#
# Tune ``C`` before drawing conclusions about strategy preference.

# %%
# References
# ----------
#
# .. [#1] `"On the Algorithmic Implementation of Multiclass Kernel-based
#    Vector Machines." Koby Crammer and Yoram Singer.
#    Journal of Machine Learning Research 2 (2001): 265–292.
#    <https://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf>`_
