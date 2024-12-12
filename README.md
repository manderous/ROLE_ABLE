# Reasoning-Oriented and Analogy-Based Methods for Locating and Editing in Zero-Shot Event-Relational Reasoning (ROLE and ABLE)

This repository contains the implementation for the research work published at **COLING 2025** (The 31st International Conference on Computational Linguistics).

<!-- [![Colab MEMIT Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kmeng01/memit/blob/main/notebooks/memit.ipynb) -->

## Table of Contents

- [ROLE](#ROLE)
- [ABLE](#ABLE)

## ROLE
### Reasoning-oriented locating method
```
python ./experiments/ROLE_locating.py
```
Analyze locating results:
```
python ./experiments/ROLE_results_analysis.py
```
### Reasoning-oriented editing method
For encoder's MLP module:
```
python ROLE_editing_encoder.py
```
For decoder's Cross-attention module:
```
python ROLE_editing_decoder.py
```

## ABLE
```
python ABLE.py
```
Analysis of the analogicality of location
```
python ./experiments/ABLE_anal_location.py
```
Analysis of the analogicality of editing magnitude
```
python ABLE_anal_edit.py
```
