# Spatiotemporal deep learning for differential renal function estimation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24615750.svg)](https://doi.org/10.5281/zenodo.24615750)

Code for the article *Spatiotemporal Deep Learning for Differential Renal Function Estimation
from Dynamic Renal Scintigraphy: A Comparative Study of 2D and 3D Convolutional Networks*
(manuscript in preparation; the reference will be added after publication).

The code estimates the differential renal function (DRF), the percentage of total renal
function attributed to the left kidney, directly from the raw posterior projection of
<sup>99m</sup>Tc-MAG3 dynamic renal scintigraphy. Three representations of the time course are
compared:

| Representation | Input | Network |
|---|---|---|
| 2D collapsed | sum of minutes 1 to 20 | VGG19 pre-trained on ImageNet |
| Temporal mosaic | nine consecutive intervals tiled in a 3 × 3 grid | VGG19 pre-trained on ImageNet |
| 3D sequence | 19 one-minute frames | R3D-18 pre-trained on Kinetics-400 |

As a baseline without deep learning, a simplified automatic estimate is computed on the same
studies: the DRF from two square regions of interest over the kidneys in the posterior frames
from 60 to 150 s, without background correction, linearly recalibrated on the four non-test
folds of each split. Evaluation uses five-fold patient-level cross-validation with fixed
partitions, a validation fold separate from the test fold, and four seeds.

> Research code. It is not a medical device and must not be used for clinical decisions.

## Notes

- Identifiers, comments and printed messages are in Portuguese. This README and the glossary at
  the end explain the structure in English.
- Part of the code was written with the help of an AI assistant (Claude, Anthropic) and
  reviewed by the authors, as declared in the article.
- No data, trained weights or patient-level results are included.

## Requirements

Python 3.14 and an NVIDIA GPU for training. The runs reported in the article used an NVIDIA
GeForce RTX 4060 Laptop GPU (8 GB) and, for R(2+1)D-18, an NVIDIA A100 on Google Colab.

Install PyTorch and torchvision with CUDA support following https://pytorch.org, then the rest:

    pip install -r requirements.txt

`requirements.txt` lists the exact versions used (PyTorch 2.14.1, torchvision 0.29.1,
scikit-learn 1.9.1, NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1, pydicom 3.0.2, among others). The
fixed partitions are stored in the code, so other versions of scikit-learn do not change the
folds.

## Data

The data are not distributed with this code. They come from the Database of Dynamic Renal
Scintigraphy (www.dynamicrenalstudy.org), under the terms of its license agreement. Download the
DICOM package and extract it at the root of this repository, so that these folders exist:

    dynamicrenalstudy_DICOM_package/database/DATA_DICOM/   one folder per study, multi-frame DICOM
    dynamicrenalstudy_DICOM_package/database/TABLES_b/     tables with the reference values

Sets used: drsprg (107 adults) and drsbru (100 children) for training and evaluation, drshyb
(19 hybrid studies, two variants each) for training only, and one simulated study (drslund)
only to draw Fig. 1. Every intermediate file is written to `data/`, which is excluded from git
because it holds study-level information derived from the database.

## Pipeline

Run every command from the root of the repository.

### 1. Data preparation (CPU)

    python src/build_inventory.py      # inventory of the DICOM files -> data/inventory_files.csv
    python src/parse_tables.py         # reference tables -> data/labels/
    python src/build_master.py         # one row per study -> data/master.csv
    python src/eda_sanity.py           # raw ROI estimate -> data/sanity_lk_pct_cru.csv
    python src/make_colab_dataset.py   # 60-s blocks of every study -> data/colab_renal_lk.npz

`eda_sanity.py` is required, because `make_colab_dataset.py` reads its raw ROI estimate. It
also writes sanity figures of real studies to `reports/figures/`, which is excluded from git.
The `.npz` file holds 245 samples: the 207 patient studies and the 38 hybrid variants, flagged
so that they never enter validation or test.

### 2. Training (GPU)

`src/fase3_lib.py` holds the whole protocol: fixed partitions, input representations,
augmentation, models, training loop, internal validation and evaluation. `src/fase3_rodar.py`
trains one configuration from the command line, and `src/fase4_fila.py` runs the queues of the
final round with the seeds 5042, 6042, 7042 and 8042:

    python src/fase4_fila.py --etapas validacao_interna   # main results of the article
    python src/fase4_fila.py --etapas final colapsado     # first runs (sensitivity analysis)
    python src/fase4_fila.py --etapas ablacao             # ablation of the 3D configuration

- `validacao_interna`: the three methods with internal validation, in which fold (f + 1) mod 5
  selects the learning-rate reduction, early stopping and best epoch, and fold f, the test
  fold, is predicted with the selected checkpoint (60 trainings, about 4.5 h on the RTX 4060).
  The error on fold f is also recorded at each epoch, for diagnosis only; no training choice
  uses it.
- `final colapsado`: the same methods in the first runs, in which the test fold also served as
  the validation set.
- `--simular` prints the plan without training. Interrupted queues resume from the last
  completed fold. `--esperar-pid` (wait for another process to finish before starting) works
  only on Windows.

Results are written to `data/fase4/`. Other stages of the same script (`triagem`,
`atribuicao`, `grades`, `final_grades`) are the preliminary refinements of the mosaic and of the
number of frames.

### 3. R(2+1)D-18 on Google Colab

    python src/gerar_notebook_fase3.py --fase4   # writes notebooks/fase4_colab.ipynb

The notebook embeds `src/fase3_lib.py`. Copy `data/colab_renal_lk.npz` to the root of Google
Drive, open the notebook in Colab with a GPU runtime and run all cells. The notebook trains
R(2+1)D-18 and repeats the mosaic with the same seeds, because results obtained on different
GPUs are compared only with each other. Copy the output folder, `MyDrive/fase4_colab/ambos/`,
to `data/fase4_colab/ambos/`. The R(2+1)D-18 runs reported in the article used version 3.6.0
of the library; version 3.7.0, embedded here, only adds the optional internal validation, which
is off by default.

### 4. Analysis and figures

    python src/fase4_analise.py    # metrics, bootstrap intervals and tables -> data/fase4/fase4_tabelas.md
    python src/artigo_figuras.py   # Figs. 1 to 4 of the article -> artigo/figures/

## Tests

    python src/fase3_testes.py         # protocol and queue rules on the CPU (about 1 to 2 min)
    python src/fase3_testes.py --gpu   # plus one short training epoch per model on the GPU

The tests run in sequence and stop at the first failure. Most of them use synthetic data. A
few download the pre-trained torchvision weights on the first run, and `--gpu` needs
`data/colab_renal_lk.npz`.

## Files

| File | Role |
|---|---|
| `src/build_inventory.py` | inventory of the DICOM files of every set |
| `src/parse_tables.py` | reference values from the tables of the database |
| `src/build_master.py` | master table, one row per study |
| `src/eda_sanity.py` | DICOM reading, coarse kidney ROIs and raw ROI-based DRF |
| `src/make_colab_dataset.py` | 60-s blocks of every study in a single `.npz` file |
| `src/fase3_lib.py` | protocol, models, training and evaluation |
| `src/fase3_rodar.py` | training of one configuration from the command line |
| `src/fase3_investida.py` | final round of the exploratory 3D phase and helpers of the final queue |
| `src/fase4_fila.py` | queues of the final round: arms, seeds and comparison rules |
| `src/fase4_analise.py` | metrics, bootstrap confidence intervals and tables |
| `src/artigo_figuras.py` | figures of the article |
| `src/gerar_notebook_fase3.py` | builds the Colab notebook from the library |
| `src/fase3_testes.py` | tests of the library and of the queue rules |
| `notebooks/fase4_colab.ipynb` | R(2+1)D-18 and its mosaic control on Colab |

## Glossary

| Term in the code | Meaning |
|---|---|
| fase 2, 3, 4 | project phases: 2 = selection of the 2D network (VGG19 on the mosaic), 3 = exploratory 3D models, 4 = final round reported in the article |
| fila, etapa | queue of training runs, stage of a queue |
| braço, controle | arm of a comparison; the control arm is the mosaic |
| semente | random seed |
| mosaico, colapsado | temporal mosaic; collapsed (summed) image |
| validação interna (`_vi`) | fold (f + 1) mod 5 selects the training choices; fold f is the test fold |
| triagem | screening with two seeds and three folds |
| cabeça espacial | spatial head of the 3D network |
| aquecimento | learning-rate warm-up |
| saída na média (`_sm`) | final layer initialized at the mean of the target |
| lk, lk_cru | reference DRF (% left kidney); raw ROI-based estimate |
| PDI crua, PDI recal | automatic ROI-based estimate (the baseline), raw and recalibrated |
| híbridos (drshyb) | hybrid studies with known DRF, used for training only |

## Citation and license

Please cite the article (reference to be added after publication) and the archived version of
this code on Zenodo. The article cites version 1.0.0 (https://doi.org/10.5281/zenodo.24615751);
the DOI 10.5281/zenodo.24615750 always resolves to the latest version. `CITATION.cff` has the
metadata. The code is released under the MIT License (see `LICENSE`). The license covers the
code only; the data remain under the license agreement of the Database of Dynamic Renal
Scintigraphy.
