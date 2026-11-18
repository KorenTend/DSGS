<div align="center">

<img src="assets/readme-hero.svg" alt="DSGS — Dental Mesh Simplification" width="100%" />

<h2>DSGS: Dental Mesh Saliency-Guided Simplification<br />with Geometric Feature Preservation</h2>

**Ke Li · Yutong Hu · Haisheng Li · Li Chen**

Saliency estimation and geometric processing for dental triangle meshes.

[Installation](#installation) &nbsp; · &nbsp;
[Quick start](#quick-start) &nbsp; · &nbsp;
[Configuration](#configuration) &nbsp; · &nbsp;
[Code](#repository-structure) &nbsp; · &nbsp;
[Citation](#citation)

</div>

---

## Installation

Use **Python 3.10** and install the **PyTorch 2.5.1** build appropriate for your
hardware. Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

<details>
<summary><strong>CPU installation</strong></summary>

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

</details>

The code has been checked on Windows. PyMeshLab and Open3D may require
platform-specific system libraries on other systems.

## Quick start

The `paper` profile trains a DSN for the input mesh and simplifies it at the
chosen face-retention ratio. The command uses the experiment parameters
`BETA`, `ETA`, `R0` and `GAMMA`.

```bash
python main.py -i data/your_mesh.stl -o output/DSGS_50.obj -r 0.5 --beta "$BETA" --eta "$ETA" --r0 "$R0" --gamma "$GAMMA" --device auto
```

`-r 0.5` requests **50% of the cleaned input faces**. The output metadata
records the target and actual face counts.

The height feature assumes **positive z points toward the occlusal surface**.
Patch coordinates are centered and scaled for training. The exported mesh uses
the original coordinate system. Each input mesh is trained independently.

### Reuse a checkpoint

Train once and reuse the checkpoint for another face budget on the same mesh:

```bash
python main.py -i data/your_mesh.stl -o output/DSGS_30.obj -r 0.3 --checkpoint output/DSGS_50.pth --eta "$ETA" --r0 "$R0" --gamma "$GAMMA" --device auto
```

The checkpoint restores the training profile, patch settings and seed. Use the
same input mesh and protection parameters to compare face budgets. A new output
stem keeps each run's metadata together with its mesh.

Checkpoints from the earlier release automatically use the `legacy` profile.
Select `--profile legacy` to train with the earlier architecture, losses and
PyMeshLab plus surface-fitting workflow.

## Configuration

| Option | Default | Purpose |
| :--- | :---: | :--- |
| `--ratio`, `-r` | `0.1` | Fraction of faces retained, in `(0, 1]` |
| `--device`, `-d` | `auto` | Select `auto`, `cpu` or `cuda` |
| `--epochs` | `100` | Training epochs per input mesh |
| `--num-patches` | `16` | Number of candidate patches |
| `--patch-points` | `4096` | Sampled points per patch |
| `--batch-size` | `4` | Patches per training step in the paper profile |
| `--seed` | `42` | Random seed |
| `--profile` | `paper` for new training | Select the manuscript or legacy implementation |
| `--beta` | — | Curvature coefficient in the virtual collapse mask, Eq. (12) |
| `--eta` | — | Exponential branch denominator, Eq. (16) |
| `--r0` | — | Retention threshold in `(0, 1)`, Eq. (16) |
| `--gamma` | — | Constant multiplier at and above `r0`, Eq. (16) |
| `--checkpoint` | — | Load a checkpoint and skip training |

Defaults are defined in [config.py](config.py). For all command-line options:

```bash
python main.py --help
```

## Repository structure

| File | Role |
| :--- | :--- |
| [main.py](main.py) | Training and simplification entry point |
| [model.py](model.py) | Saliency network |
| [loss.py](loss.py) | Training objectives |
| [config.py](config.py) | Default configuration |
| [utils_geo.py](utils_geo.py) | Geometric features and patch processing |
| [runtime.py](runtime.py) | Mesh loading, seeds and input validation |
| [simplify.py](simplify.py) | Saliency-weighted edge-collapse QEM and legacy decimation |
| [surface_refinement.py](surface_refinement.py) | Surface fitting in the legacy profile |

## Outputs

For `--output output/DSGS_50.obj`, the run produces:

| Artifact | Contents |
| :--- | :--- |
| `DSGS_50.obj` | Simplified mesh |
| `DSGS_50.pth` | Trained checkpoint, when training is performed |
| `DSGS_50_weights.npz` | Stitched vertex saliency and curvature |
| `DSGS_50.json` | Configuration, versions, hashes, face counts and timing |

## Manuscript implementation

- Eq. (2) uses the magnitude of the vertex angle deficit as the discrete Gaussian
  curvature prior, following the text's use of curvature magnitude. The scores
  are clipped at the mesh-wide 50th and 90th percentiles, normalized, and raised to
  the power `0.5`.
- Eqs. (5)–(10) use four dynamic EdgeConv layers and a learned scalar gate for
  each vertex before the saliency and folding branches. The curvature blend
  starts at `alpha=2.0`.
- Eqs. (9), (11)–(13) use Charbonnier reconstruction, virtual collapse with eight
  point neighbors, and mean saliency. Their weights are `1.0`, `1.0` and `2.0`,
  respectively.
- Eqs. (14)–(16) apply a boundary override of `10.0` and use the maximum endpoint
  saliency in the edge priority. The retention multiplier is
  `exp((1 - r) / eta)` below `r0` and `gamma` at or above it.

The implementation retains original boundary coordinates and edges. Each merged
vertex inherits the larger endpoint saliency. A singular quadric or a position
that would flip an incident face uses the lowest-cost valid endpoint or midpoint.
Patch fusion leaves unsampled vertices at zero before global min–max normalization.

## Citation

Use [CITATION.cff](CITATION.cff) when citing the software.

## License

Released under the [Apache License 2.0](LICENSE). Third-party dependencies retain
their respective licenses.
