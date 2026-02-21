"""structflo.cser.lps — Learned Pair Scorer for structure-label association.

Public API::

    from structflo.cser.lps import LearnedMatcher

    # Use as a drop-in replacement for HungarianMatcher
    from structflo.cser.pipeline import ChemPipeline
    pipeline = ChemPipeline(matcher=LearnedMatcher("runs/lps/scorer_best.pt"))

Training::

    sf-train-lps --data-dir data/generated --epochs 30

Evaluation::

    sf-eval-lps --weights runs/lps/scorer_best.pt
"""

from structflo.cser.lps.matcher import LearnedMatcher

__all__ = ["LearnedMatcher"]
