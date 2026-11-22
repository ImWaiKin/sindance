# SinDance: dance exergame movement features, postural balance and cognitive performance

Code for the manuscript *A video-based dance exergame to estimate postural balance and cognitive
performance in community-dwelling older adults: a cross-sectional proof-of-concept study*
(JMIR Serious Games, ms#107960, revision 1).

No participant data are stored in this repository. De-identified keypoint feature data may be made
available from the corresponding author on reasonable request, subject to institutional and ethical
approvals.

## Repository layout

| Folder | Contents |
|---|---|
| `1. Feature Extraction from Dance Raw Data/` | Pre-processing of MediaPipe keypoint logs and extraction of the 77 dance movement features |
| `analysis/` | All analyses reported in the revised manuscript (balance label, balance classification, cognitive estimation, missing data) |
| `analysis/results/` | Aggregate outputs of the analyses (no participant-level data) |
| `archive/original_submission/` | Notebooks used for the original submission, kept for transparency; superseded (see its README) |

## Data processing

**Keypoints.** During gameplay, MediaPipe Pose estimates 33 landmarks per frame in real time. Video
frames are discarded after processing; only time-stamped landmark coordinates are stored. Facial
landmarks 0-10 are not stored (saved as -1000), leaving 22 body landmarks. Across the 81 recordings,
landmarks were stored at a median of 56 frames per second (IQR 53-57; range 33-58) for about 95 s of
the dance.

**Pre-processing** (`2. data_preprocessing.py`): missing landmarks (-1000) are set to missing;
detections with confidence <= 0.5 are removed; for each participant, coordinates outside the 1st-99th
percentile are removed as tracking errors; timestamps are reset to start at 0. Coordinates are
MediaPipe's normalised image coordinates (0-1). No temporal smoothing or interpolation is applied.

**Features** (`4. advanced_feature_extraction.py`): 77 features covering temporal, frequency-domain,
symmetry, stability, coordination, sway and interaction measures (listed in Multimedia Appendix 2).

## Analyses (`analysis/`)

Put the data in `data/` at the repository root (or set `SINDANCE_DATA`):
`hur_performance.csv`, `enhanced_features.csv`, `clinical.csv` (see `analysis/common.py`).

```bash
pip install -r requirements.txt
cd analysis
python consolidate_hur_data.py <folder of HUR .xls exports>   # -> data/hur_performance.csv
python balance_labels.py      # composite balance score and lower/higher composite sway groups
python balance.py             # balance classification (about 12 min)
python cognitive.py           # MoCA estimation
python missing.py             # Little's MCAR test and multiple imputation
```

| Manuscript item | Script | Output |
|---|---|---|
| Balance label: formula, weights, 0.33 cut-point | `balance_labels.py` | `data/balance_labels.csv` |
| Balance classification: LOOCV ensemble (BRF + SVM, SMOTE in-fold), no-augmentation baseline, nested cross-validation, permutation test, calibration, repeated 5-fold CV, cut-point and continuous-outcome sensitivity analyses, Figure 5 | `balance.py` | `results/balance_results.txt`, `results/figure5_roc_balance.png`, `results/calibration_balance.png` |
| Cognitive estimation: five predictor sets on identical LOOCV splits (ridge regression; random forest sensitivity analysis), null model, paired comparisons, permutation test | `cognitive.py` | `results/cognitive_results.txt` |
| Missing data: Little's MCAR test, comparison of participants with and without platform data, multiple imputation (m=20) | `missing.py` | `results/missing_data_results.txt` |

### Balance label
The composite balance score is described in full in `balance_labels.py`. The condition weights,
metric weights and 0.33 cut-point were set by the authors. Normalisation (median, MAD, winsorising
limits, minimum and maximum) uses this sample, so the score describes relative standing within the
sample, is not a clinical threshold, and cannot be applied to a new individual without fixed
reference values. Because each trial lasts 30 s, Trace Length and Mean Sway Velocity are the same
quantity.

### Reproducibility
Results were produced with Python 3.9 and the package versions in `requirements.txt`. Other versions
of scikit-learn or imbalanced-learn can change individual predictions; for example, the original
submission reported 48/77 correct balance classifications, whereas the same code gives 47/77 with
these versions.
