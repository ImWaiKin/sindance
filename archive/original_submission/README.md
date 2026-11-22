# Original submission notebooks (superseded)

These notebooks produced the results in the original submission. They are kept unchanged for
transparency. The revised manuscript uses the scripts in `analysis/` instead, because peer review
identified the following problems, confirmed by re-running these notebooks:

- **Cognitive Model 1** (`Cognitive Model A - All Data`): evaluated on a single 85/15 train-test split
  (13 test participants), not leave-one-out cross-validation. The reported R², MAE and tolerance
  percentages (e.g. 9/13 = 69.2%) came from those 13 participants. The model reported was a voting
  ensemble, not the SVR described in the original Methods.
- **Cognitive Model 2** (`Cognitive Model B - Cognitive and Dance Only`): the NNLS ensemble weights,
  the choice of component models and the choice of best model were fitted on the outer LOOCV
  predictions, i.e. using the test outcomes.
- **Both cognitive models** used more predictors than were listed (including weight, height, recent
  falls, activity level, marital status, employment and ethnicity) and coded education
  alphabetically rather than ordinally.
- **Balance model** (`3. Balance_Output_Prediction.ipynb`): the augmentation strategy and
  hyperparameters were chosen on the same LOOCV results that were reported; the reported AUC
  (0.752) came from the Balanced Random Forest alone rather than the reported ensemble; the isotonic
  calibration used a single split with 13 participants in the calibration set. Code comments in this
  notebook referring to other accuracies (e.g. Mixup 75.3%) are out of date.

A saved trained model and the decision-curve figure were removed from the repository: the model
file stored participant-level training values, and decision curve analysis is no longer reported.
`run_notebook.py` executes a notebook's code cells unchanged (used to confirm the numbers above).
