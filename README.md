# BrainML
# Cognitive Workload Detection from fNIRS Brain Blood Flow

Predicting how hard someone's brain is working (e.g., 0-back vs 2-back vs 3-back
tasks) from fNIRS signals, with a focus on generalizing to people the model has
never seen.

## Status
🚧 Work in progress. Currently: setting up data loading and preprocessing.

## Goal
Build a model that classifies mental workload level from prefrontal fNIRS
(HbO / HbR) signals, and evaluate it on unseen subjects.

## Dataset
- TU Berlin hybrid EEG-fNIRS dataset (n-back task) [add link]

## Planned approach
1. Load and explore the data
2. Preprocess: filtering, motion artifact correction, short-channel regression
3. Baseline: simple features + logistic regression / SVM
4. Deep learning model (CNN / EEGNet-style)
5. Leave-subject-out evaluation
6. Interpretability: which channels and time windows matter

## Results
Coming soon.

## Project structure
(to be added as the project grows)

## Setup
(to be added)
