# Heart-Rate Estimation from Wrist PPG During Motion (PPG-DaLiA)

The goal of this project is to estimate heart rate from wrist PPG during real-world activity. The dataset chosen is PPG-DaLiA, a 15-subject dataset with wrist and chest signal data. Accelerometer and PPG data are taken from the wrist, while the reference heart rate used in this study is pre-computed from a single-lead chest ECG.

I compared two motion-aware classical signal processing methods against a 1D CNN, evaluating all three under the same leave-one-subject-out framework so that the results are directly comparable.

## Results

| Method | Mean MAE ± SD (bpm) | Median subject MAE (bpm) |
|---|---|---|
| Accelerometer-vetoed spectral peak | 13.01 ± 8.14 | 9.84 |
| Simplified spectral subtraction | 12.94 ± 7.84 | 10.10 |
| 1D CNN, nested LOSO | **8.84 ± 5.19** | **6.94** |

The ± value is the standard deviation across the 15 held-out subjects. I reported subject-level means rather than a pooled figure so that the subjects with longer recordings do not dominate the result.

## Repository contents

The analysis is contained in a single notebook, `ppg-analysis-portfolio-nested-training.ipynb`. The `results` directory holds `cnn_nested_mae.json` with the per-subject held-out MAE for the 15 CNN folds, `cnn_nested_manifest.json` with the fold splits, seeds, selected epochs and normalization statistics, and `cnn_nested_predictions.npz` with the prediction and reference arrays for every subject.

I committed these three aggregate files so that the figures and comparison tables can be reviewed without retraining. The dataset itself, the fold checkpoints and the per-fold prediction files are not tracked.

## Setup

PPG-DaLiA is not redistributed here. It can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/495/ppg+dalia) and unpacked so that the `PPG_FieldStudy` directory contains one folder per subject, each holding a single pickle file from `S1/S1.pkl` through `S15/S15.pkl`.

I used the directory structure provided with PPG-DaLiA. The notebook searches `PPG_DALIA_ROOT`, then `../data/PPG_FieldStudy`, `data/PPG_FieldStudy` and `./PPG_FieldStudy` in that order. Set `PPG_DALIA_ROOT` to the `PPG_FieldStudy` directory if the dataset is stored somewhere else.

```bash
export PPG_DALIA_ROOT=/path/to/PPG_FieldStudy
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib torch jupyter
jupyter notebook ppg-analysis-portfolio-nested-training.ipynb
```

I developed this on Python 3.12 with NumPy 2.2, SciPy 1.15 and PyTorch 2.13. PyTorch is imported inside the CNN section rather than at the top, which keeps the classical sections usable without it.

Setting `RUN_CNN = True` reruns the full CNN LOSO experiment; otherwise the notebook uses the subject-level MAEs from my completed run. Setting `RESUME_CNN = True` skips any fold whose checkpoint and prediction files already exist. Training uses CUDA, Apple MPS or CPU, selected automatically. Each outer fold trains twice, once to select the epoch count and once to fit the final model, so a complete run takes considerably longer on CPU. Progress is written to disk after every fold, so an interrupted run can be resumed.

## Loading the data

I loaded each subject's wrist PPG at 64 Hz and three-axis accelerometer at 32 Hz along with the ECG-derived heart-rate labels, which are provided at 2-second intervals. The notebook confirms that all 15 subject files are present before running the analysis. The subjects contribute roughly 60,700 windows in total, with reference heart rate spanning 41.7 to 187.0 bpm across the cohort.

## Motion peaks within the heart-rate band

I divided the PPG and accelerometer signals for each subject into overlapping 8-second windows with a 2-second shift. For each window I then took the FFT of the signal. Peak detection was used to identify the dominant frequencies in the signals.

When pooling all of the accelerometer peaks across windows and subjects into a single distribution, I found that 68% of the 265,206 peaks fell within the heart-rate frequency band of 0.5 to 4 Hz. Motion energy is therefore not separable from cardiac energy by frequency alone, which is why the accelerometer is used as side information in the methods below rather than as a simple filter.

## Method 1: Accelerometer-vetoed spectral peak detection

Starting with subject 1 (S1), I applied a spectral peak detector to the FFT of its windows and, for each window, picked the strongest PPG peak that was not near the accompanying accelerometer spectral peak or peaks. I then converted the frequency of that peak to heart rate and used median filtering to suppress isolated outlier estimates.

The PPG was bandpass filtered between 0.5 and 4 Hz with a fourth-order zero-phase Butterworth filter before the transform. Accelerometer peaks were detected on the mean-removed magnitude signal using a height threshold of the spectral mean plus one standard deviation, while PPG peaks were detected on a prominence threshold and ranked by prominence. When every PPG candidate was vetoed, the estimator fell back to the most prominent peak.

To apply the method across all subjects, I selected the accelerometer-veto tolerance and median-filter kernel using only the 14 training subjects in each LOSO fold. The tolerance was searched over 0.10, 0.15 and 0.20 Hz and the kernel over 7, 11, 15 and 21. Every fold selected the widest kernel and the tightest tolerance.

I then applied this technique to all subjects using an LOSO framework for proper cross-validation, achieving **13.0 ± 8.1 bpm MAE** with a median subject MAE of **9.8 bpm**.

## Failure analysis for Method 1

To understand where this method failed, I analyzed windows where the raw estimate was more than 10 bpm off before median filtering. I grouped each error according to whether the true PPG peak was absent, rejected as motion, or present but not selected because a stronger competing peak was chosen.

| Failure mechanism | Share of raw failures |
|---|---|
| True-frequency peak absent | 49.5% |
| Competing peak selected | 27.7% |
| True-frequency peak vetoed | 21.3% |
| All candidates vetoed | 1.5% |
| No PPG peaks | 0.0% |

Close to half of the failures were windows in which the correct peak was not present in the spectrum at all, so no improvement to the peak selection rule could have recovered them. This is part of the motivation for a method that does not depend on a visible spectral peak.

I then analyzed median filtering separately because the filtered value can be affected by neighboring windows rather than only the peak decision made in the current window. Smoothing corrected 18.3% of all windows and introduced new errors in 8.7%, leaving 50.2% correct before and after and 22.8% incorrect in both cases.

## Motion-spectrum subtraction

To improve the classical result, I tried a simplified SpaMaPlus-style spectral subtraction approach taken from the literature. This approach subtracts the scaled accelerometer spectrum from the PPG spectrum using

$$S_{\mathrm{clean}} = S_{\mathrm{PPG}} - \alpha S_{\mathrm{accel}}$$

where $\alpha$ is chosen to minimize the mean MAE across the training subjects during each LOSO fold. The accelerometer spectrum was interpolated onto the PPG frequency grid and rescaled to the peak magnitude of the PPG spectrum before subtraction. I took the argmax of the cleaned spectrum inside a physiological mask running from 40 bpm to an age-based maximum of 220 minus the subject age from the dataset questionnaire, and applied a median filter with a fixed kernel of 21.

This approach tied the first baseline, achieving **12.9 ± 7.8 bpm MAE** with a median subject MAE of **10.1 bpm**. LOSO tuning consistently selected the largest tested $\alpha$ of 2.0, showing that removing more accelerometer spectral energy produced better training-subject performance within this implementation.

## 1D CNN

I then trained a 1D CNN using four standardized time-domain channels: PPG and the three accelerometer axes. The accelerometer channels were linearly interpolated onto the PPG timestamps, and each window was normalized per channel with a z-score so that the model could not rely on absolute amplitude differences between subjects. Each window contained 8 seconds of data, giving 512 samples per channel, and the outer evaluation remained LOSO, with one subject held out for testing in each fold.

The network applies four convolutional blocks, each consisting of a 1D convolution, batch normalization, a ReLU and max pooling, with channel widths of 32, 64, 128 and 128 and kernel sizes of 7, 5, 5 and 3. Adaptive average pooling is followed by a regression head with 30% dropout, a 128 to 64 linear layer, a ReLU and a final linear output. I trained with AdamW at a learning rate of 3e-4 and weight decay of 1e-4, a Smooth L1 loss with a beta of 0.5, a batch size of 512 and a plateau scheduler that halves the learning rate after two epochs without improvement. Targets were standardized using the mean and standard deviation of the outer training fold.

For each outer fold, I used two of the remaining subjects as an inner validation set. The validation subjects were used to choose the number of training epochs through early stopping, with a maximum of 40 epochs, a minimum of 8, a patience of 6 and a minimum improvement of 0.02 bpm. I then initialized a new model, retrained it on all 14 non-test subjects for the selected number of epochs, and evaluated it once on the held-out subject. This prevents the held-out subject from influencing model selection while still allowing the final fold model to use all available training subjects.

The selected epoch count ranged from 2 to 17 across the 15 folds, so a single fixed epoch budget would have been wrong for most of them. Each outer fold saves its final model weights, target normalization, subject split, seed, selected epoch, validation history and training configuration.

Under this procedure the CNN reached **8.84 ± 5.19 bpm MAE** with a median subject MAE of **6.94 bpm**.

## Note on the earlier CNN run

An earlier version of this project trained every fold for a fixed number of epochs without an inner validation split and reported 7.69 ± 4.29 bpm. Those values are retained in the notebook as a clearly labeled historical comparison and appear in the table below for reference. The nested procedure described above is the one the committed results come from, and it reports the higher figure of 8.84 bpm.

The increase is expected. The earlier run used an epoch budget that was never validated against data independent of the held-out fold, so its result was optimistic. The nested figure is the one I would cite.

## Subject-level comparison

I placed the results from all three methods into one subject-level comparison table so that the mean result does not hide differences between held-out subjects.

| Subject | Accelerometer-veto MAE | Spectral-subtraction MAE | CNN MAE, nested | CNN MAE, earlier run |
|---|---|---|---|---|
| S1 | 7.17 | 7.10 | 6.82 | 5.8 |
| S2 | 9.42 | 9.60 | 6.39 | 5.5 |
| S3 | 10.24 | 10.05 | 5.17 | 4.0 |
| S4 | 15.29 | 14.63 | 7.13 | 6.9 |
| S5 | 39.48 | 39.42 | 26.57 | 18.4 |
| S6 | 17.74 | 16.33 | 6.62 | 7.5 |
| S7 | 4.79 | 5.91 | 5.50 | 3.6 |
| S8 | 9.79 | 9.50 | 11.13 | 16.6 |
| S9 | 17.63 | 14.65 | 11.91 | 10.8 |
| S10 | 9.84 | 11.18 | 7.27 | 5.1 |
| S11 | 18.29 | 18.36 | 9.70 | 8.5 |
| S12 | 9.69 | 8.02 | 10.44 | 8.3 |
| S13 | 7.58 | 10.10 | 6.38 | 4.2 |
| S14 | 8.08 | 8.55 | 4.67 | 4.8 |
| S15 | 10.09 | 10.72 | 6.94 | 5.4 |

Eight of the 15 subjects fall below 7 bpm under the CNN, while S5 alone contributes more than a fifth of the summed error. S5 is also the hardest subject for both classical methods, at close to 39 bpm in each case.

## Subject diagnostics

I compared CNN MAE with each subject's heart-rate distribution, PPG RMS amplitude and high-pass accelerometer RMS amplitude to investigate why the error was concentrated in a small number of subjects.

The diagnostics suggest that S5 was difficult partly because its reference heart-rate distribution was centered near 126 bpm, well above the rest of the cohort, placing it in a sparsely represented region of the training targets. S8 sat among the lowest PPG RMS amplitudes in the cohort, so the signal was weak before any processing began. Accelerometer RMS was more similar across subjects, so motion magnitude alone did not explain the high-error tail.

## Reproducibility

I set a global seed of 42, a per-fold seed of 42 plus the fold index for inner epoch selection, and that value plus 10,000 for the final training run. Deterministic algorithms are requested from PyTorch where they are supported.

The manifest records, for every fold, the outer training subjects, the inner optimization and validation subjects, both seeds, the selected epoch count, the best inner validation MAE, the target normalization statistics and the final held-out MAE. Each saved checkpoint additionally stores the full model and training configuration, the per-epoch validation history, the preprocessing settings and the library versions used.

## Limitations and conclusions

The CNN error was concentrated in four of the 15 subjects: S5 at 26.6 bpm, S9 at 11.9 bpm, S8 at 11.1 bpm and S12 at 10.4 bpm, while 8 of the 15 subjects were below 7 bpm MAE. Overall, the two motion-aware spectral methods remained near 13 bpm mean subject MAE, while the 1D CNN reduced the error to 8.84 bpm under the same LOSO framework. The main remaining limitation is uneven cross-subject performance rather than average performance across the easier subjects.

The reference heart rate is itself derived from a chest ECG rather than measured beat by beat at the wrist, so label noise is present in both training and evaluation. The two classical methods are deliberately simple reimplementations rather than tuned reproductions of published systems, and both grid searches saturated at an endpoint of their parameter range, which suggests the ranges were too narrow. Each fold was also run with a single seed, so the fold-level variance introduced by initialization is not quantified here.

### Reference

Reiss, A., Indlekofer, I., Schmidt, P., and Van Laerhoven, K. *Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks.* Sensors, 2019, 19(14), 3079.
