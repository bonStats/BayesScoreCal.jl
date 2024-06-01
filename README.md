# BayesScoreCal.jl: Bayesian Scoring Rule Calibration

[![Build Status](https://github.com/bonStats/BayesScoreCal.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bonStats/BayesScoreCal.jl/actions/workflows/CI.yml?query=branch%3Amain)

An extensible implementation of the methods described in [Bayesian score calibration for approximate models](https://arxiv.org/abs/2211.05357)

- Code demonstrating [BSRC for approximate models](https://arxiv.org/abs/2211.05357), see [BayesScoreCalExamples.jl](https://github.com/bonStats/BayesScoreCalExamples.jl).
- This package is extensible by implementing other transformations, see `/transforms/cholesky-affine.jl` for example.