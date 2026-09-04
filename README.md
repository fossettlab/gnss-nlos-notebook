# gnss-nlos-notebook

A teaching notebook that flags untrustworthy GNSS signals. Written for
**EEPS 4684 / 5684 Geospatial Field Methods** at Washington University in
St. Louis, Exercise 02.

**[Open it in Google Colab](https://colab.research.google.com/github/fossettlab/gnss-nlos-notebook/blob/main/ex02_part4_nlos.ipynb)** — nothing to install.

## What it does

A position solver normally treats every satellite's signal as equally
trustworthy. Near buildings or under canopy a signal can reach the antenna only
after bouncing off a surface, arriving late and making the satellite look farther
away than it is. That is multipath; a signal with no clean direct path at all is
called non-line-of-sight, or NLOS. A position built from bounced signals is
pulled off, often by metres.

The notebook reads a RINEX observation file and predicts, for every satellite at
every epoch, whether that signal arrived cleanly. It does not compute a position;
it flags signals.

Five features per (epoch, satellite), all from standard RINEX fields plus one
single-point-position solve via [pyrtklib](https://github.com/IPNL-POLYU/pyrtklib):
carrier-to-noise density, satellite elevation, normalized pseudorange residual,
agreement between Doppler and pseudorange rate, and short-window C/N0 variability.

The classifier is a scikit-learn random forest, trained in the notebook itself
rather than shipped pre-trained, because a pickled model only loads reliably
under the scikit-learn version that wrote it. Training takes a few seconds.
Held-out performance, averaged over leave-one-run-out folds: accuracy about 0.83,
F1 about 0.75, ROC-AUC 0.88 to 0.91.

## Contents

| File | |
|---|---|
| `ex02_part4_nlos.ipynb` | the notebook |
| `extract_features.py` | turns a RINEX observation file into the five features |
| `nlos_features.csv.gz` | the labelled training table, 26,522 observations |

## Training data and attribution

`nlos_features.csv.gz` is a numeric feature table derived from **KLTDataset**, a
GNSS collection from Kowloon Tong, Hong Kong, in which per-satellite LOS/NLOS
ground truth comes from a sky-pointing fisheye camera. It contains no imagery.

- Dataset: <https://github.com/ebhrz/KLTDataset> (GPL-3.0)
- Paper: Hu, R., Wen, W., & Hsu, L.-T. (2023). Fisheye camera aided NLOS
  exclusion and learning-based pseudorange correction. *2023 IEEE 26th
  International Conference on Intelligent Transportation Systems (ITSC)*,
  6088–6095.

The camera exists only to create the training labels. The shipped classifier
never sees an image; it predicts what a camera would have shown from radio
properties alone, because a field receiver has no camera.

**The model was trained in a dense high-rise urban canyon. Whether it
generalizes to other environments is untested**, and judging that is the point of
the exercise it was written for.

## License

GPL-3.0, inherited from KLTDataset. See `LICENSE`.
