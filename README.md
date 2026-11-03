# HYDRA

**Dynamic Materials Testing Suite** — a desktop application for processing dynamic and quasi-static material characterisation tests (Split-Hopkinson Pressure Bar, Direct Impact, quasi-static, confined quasi-oedometric), with Monte-Carlo uncertainty propagation, Digital Image Correlation, and crack / wave analysis from high-speed imaging.

> Developed and designed by Dr. Charles Francart · Current version: **v1.0**

HYDRA (`shpb_app`, Python · PySide6 · matplotlib) is a ported and extended re-implementation of a set of stand-alone processing scripts, gathered into a single graphical workflow: describe the test, load the raw signals, correct for wave dispersion, process, propagate uncertainties, correlate the images, and export everything.

A full **technical documentation** (equations as implemented, file/function traceability, known caveats) is bundled in the app and available from the Welcome tab, in [French](assets/docs/HYDRA_documentation_technique_FR.pdf) and [English](assets/docs/HYDRA_technical_documentation_EN.pdf).

---

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Building the Windows executable](#building-the-windows-executable)
- [Documentation](#documentation)
- [Project layout](#project-layout)

---

## Features

The sidebar is organised in the order of a typical test.

### Test setup

- **Test setup** — compression, tension or shear; sub-geometry (cylindrical, square/rectangular, axisymmetric, flat, notched, hat-shaped, double/single shear); test name, temperatures, loading regime (SHPB, Direct Impact, quasi-static).
- **Special tests** — settings for the quasi-oedometric (confined compression) test.
- **Input bar / Output bar / Specimen** — bar and specimen properties, each parameter carrying its own value, dispersion and distribution (normal / uniform) for Monte-Carlo propagation; raw strain-gauge signal loading, sign and offset.
- **Custom sensors** — additional sensors (e.g. laser velocimetry) with their uncertainty, multiplicative factor and curve cutting.
- **Setup profiles and session restore** — named profiles for recurring equipment; automatic restore of the last session.

### Test processing

- **Dispersion correction** — elastic (Pochhammer–Chree) and viscoelastic wave-dispersion correction for SHPB and Direct Impact, with pulse windows picked directly on the raw signals.
- **Processing** — stress / strain / strain-rate / force / temperature computation for compression, tension, notched and shear tests, in SHPB, Direct Impact and quasi-static modes. Optional physical corrections: platens, punching/indentation, inertia, friction, bar thermal gradient, adiabatic heating (Taylor–Quinney), machine compliance, and toe correction (with the pre-toe curves kept in the exports).
- **Monte-Carlo uncertainty propagation** — the whole pipeline is replayed for each random draw; mean curve and uncertainty bands are returned for every quantity.
- **Confined (quasi-oedometric) analysis** — radial stress, pressure, deviator, radial / volumetric / equivalent strains and strain rates, linear or non-linear confinement-cell models, laser-velocimetry axial stress.
- **Results & Export** — plots, comparison between tests, CSV / TXT / Excel export, and "Export full test" (rebuildable test, bar signals embedded).
- **Covariance matrix** — full Monte-Carlo covariance / correlation between points of a curve.
- **Batch analysis** — combine several processed tests into a mean curve with experimental, material and total variability bands.

### Image processing

- **Test / Video synchronisation** — locate the trigger frame of a high-speed camera from its clock file and give every frame an absolute time on the signals' time base.
- **Cameras Synchronisation** — pre-processing pipeline for **1 to 8 cameras**:
  - master dark / flat-field calibration with automatic rejection of bad frames, per-image dark scaling, pedestal and negative-pixel diagnostics;
  - rotation and translation alignment in a single cubic warp, common crop;
  - pairwise log-gain photometric equalisation between images and cameras (optional LUT tone matching, background homogenisation, denoising);
  - ImageJ-style "B&C Auto" contrast stretch;
  - time stamps (`timestamps.csv`), and lossless videos (FFV1 16-bit in MKV/AVI, H.264 lossless in MP4), merged and per camera.
- **Digital Image Correlation** — subset-based tracking with affine IC-GN sub-pixel refinement (or fast parabolic mode), ROI shapes, correlation-quality and neighbour-consistency filters, strain fields (exx, eyy, exy, von Mises, principal strains, max shear), spatial smoothing, kinematics (velocity, acceleration, dynamic force), true-strain conversion, a DIC measurement-uncertainty model, and annotated video export.
- **Crack & Wave analysis**
  - interactive or automatic (ridge-based) crack tracing, with branches and manual correction;
  - elastic wave speed from front tracking;
  - crack-tip tracking from the displacement opening jump, with wave rejection and uncertainty;
  - stress-intensity factors K<sub>I</sub> / K<sub>II</sub> by three independent methods: Williams ring fit (with optional T-stress), interaction integral, and X-FEM-DIC re-correlation of the raw images;
  - K<sub>I</sub> versus tip speed, local crack frame, and MP4 export of the analyses.

### Calibration

- **Free-decay damping** — damping of a bar from a free-decay signal.
- **Viscoelastic model calibration** — viscoelastic bar parameters from two free-decay pulses (Bayesian optimisation or fixed-parameter test), pushable to the dispersion-correction page.
- **Bayesian calibration** — Bayesian calibration of constitutive laws from free-text equations (port of MAMBA on PyMC 5 / nutpie), with covariance-library priors, posterior diagnostics and cross-validation.
- **Strain Gage Bridge Coefficient** — bridge-coefficient calculator (quarter, half, full bridges), with uncertainty, optional transverse-sensitivity and thermal-output corrections.

---

## Requirements

### Windows executable (recommended for end users)

- Windows 10 or 11, 64-bit.
- **Nothing else to install**: Python, the libraries and a C++ compiler are *not* needed — the executable is self-contained.
- `ffmpeg` is bundled inside the executable (used for the lossless **videos** of the *Cameras Synchronisation* page).

### Running from source

- Python 3.12 (the version used to develop and build the application).
- The dependencies listed in [`requirements.txt`](requirements.txt): `PySide6`, `numpy`, `scipy`, `sympy`, `matplotlib`, `opencv-python-headless`, `tifffile`, `scikit-image`, `pymc`, `pytensor`, `arviz`, `numba`, `nutpie`, `openpyxl`, `imageio-ffmpeg`.
- No separate `ffmpeg` installation is needed: a static `ffmpeg` binary comes with `imageio-ffmpeg` (an `ffmpeg` already on your `PATH` is used first, if present).
- No C/C++ compiler is required: the Bayesian calibration uses the `nutpie` sampler (Numba backend).

---

## Installation

### From source

```bash
git clone https://github.com/<your-account>/<your-repo>.git
cd <your-repo>   # the project root (HYDRA/)

python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python main.py
```

---

## Usage

1. Open the **Welcome** tab: the two buttons open the technical documentation (PDF) in French or in English.
2. Describe the test in **Test setup**, **Input bar**, **Output bar** and **Specimen**, and load the raw signals.
3. (Optional) apply the **Dispersion correction**, then run **Processing**.
4. Inspect and export in **Results & Export**; add **Covariance matrix** or **Batch analysis** if needed.
5. For tests with imaging: **Cameras Synchronisation** → **Digital Image Correlation** → **Crack & Wave analysis**.

Parameters and file paths are remembered between sessions (stored locally; nothing is sent over the network).

---

## Building the Windows executable

The executable is built with PyInstaller, directly on Windows (Python 3.12), from the project root:

```bat
python -m venv build_env
build_env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller pyinstaller-hooks-contrib

rmdir /s /q build
rmdir /s /q dist
pyinstaller --onefile --windowed --name HYDRA --icon="assets/icons/hydra_icon.ico" --add-data "assets;assets" --exclude-module PyQt5 --exclude-module PyQt5.sip --exclude-module PyQt6 --exclude-module PySide2 --collect-all pymc --collect-all pytensor --collect-all arviz --collect-all arviz_base --collect-all arviz_stats --collect-all arviz_plots --collect-all numba --collect-all nutpie --collect-all llvmlite --collect-all cv2 --collect-all skimage --collect-all imageio_ffmpeg --copy-metadata pymc --copy-metadata pytensor --copy-metadata arviz --copy-metadata arviz_base --copy-metadata arviz_stats --copy-metadata arviz_plots --copy-metadata numba --copy-metadata nutpie --copy-metadata llvmlite --copy-metadata xarray --copy-metadata pandas --copy-metadata imageio-ffmpeg main.py
```

The result is a single `dist\HYDRA.exe`. See [`BUILD_WINDOWS.md`](BUILD_WINDOWS.md) for known pitfalls (notably why no C++ compiler is needed).

---

## Documentation

| Document | Language |
|---|---|
| [Technical documentation v1.0](assets/docs/HYDRA_documentation_technique_FR.pdf) | Français |
| [Technical documentation v1.0](assets/docs/HYDRA_technical_documentation_EN.pdf) | English |

The documentation describes the equations as implemented, with `file::function` references for each, and lists the assumptions and caveats found in the code. It is a verification document, not a theory manual.

---

## Project layout

```
HYDRA/
├── main.py              # entry point
├── hydra.spec           # PyInstaller specification
├── core/                # computation (no Qt): processing, DIC, crack analysis, camera sync, calibration
├── ui/                  # PySide6 interface (pages, widgets, theme)
└── assets/              # icons, geometry/bridge drawings, documentation PDFs
```

---

## Citation and licence

If you use HYDRA in academic work, please cite the author: Dr. Charles Francart (3SR Laboratory - UGA)
