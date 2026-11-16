# DSGS: Dental Mesh Saliency-Guided Simplification with Geometric Feature Preservation

**Ke Li, Yutong Hu, Haisheng Li, Li Chen**

Implementation of DSGS for dental triangle mesh simplification.

## Installation

Use Python 3.10. Install the appropriate PyTorch 2.5.1 build for your hardware,
then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For a CPU installation:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The code has been checked in a Windows environment. PyMeshLab and Open3D may
require platform-specific system libraries on other systems.

## Code structure

```text
main.py                Training and simplification entry point
model.py               Saliency network
loss.py                Training objectives
config.py              Default configuration
utils_geo.py           Geometric features and patch processing
runtime.py             Mesh loading, seeds and input validation
simplify.py            Mesh decimation
surface_refinement.py  Surface fitting
requirements.txt       Python dependencies
```

## Input

Provide your own triangle mesh and pass its path using `--input`. STL input and
OBJ output are used in the examples below. The height feature assumes the
positive z axis points toward the occlusal surface. Coordinates are not rescaled;
use a consistent orientation and verify the physical units of the source data.

## Usage

Train on an input mesh and generate a simplified mesh:

```bash
python main.py --input data/your_mesh.stl --output output/DSGS_50.obj --ratio 0.5 --device auto
```

`--ratio` is the fraction of faces retained. A value of `0.5` requests
approximately 50% of the cleaned input faces. The target is subject to a
500-face minimum capped at the original face count; quality and boundary
constraints may affect the actual output budget.

The default configuration uses 100 epochs, 16 candidate patches and 4,096
points per patch. Each input mesh is trained independently. Use `--device cpu`
to run on CPU or `--device cuda` to select an available CUDA device.

For a short execution check on a sufficiently dense input mesh:

```bash
python main.py -i data/your_mesh.stl -o output/check.obj -r 0.5 --device cpu --epochs 1 --patch-points 128 --num-patches 2
```

These reduced settings are intended for a software check. Use the default
configuration for normal experiments. Additional options are listed by:

```bash
python main.py --help
```

## Checkpoint reuse

Reuse a trained checkpoint for another face budget on the same mesh:

```bash
python main.py -i data/your_mesh.stl -o output/DSGS_30.obj -r 0.3 --checkpoint output/DSGS_50.pth --device auto
```

Checkpoints store network state, configuration and seed. Their saved patch
settings take precedence during loading. The retention ratio and device remain
selectable. Use a new output stem to preserve the previous run's metadata.

## Outputs

For an output path such as `output/DSGS_50.obj`, the program writes:

- `DSGS_50.obj`: simplified mesh.
- `DSGS_50.pth`: trained checkpoint when training is performed.
- `DSGS_50_weights.npz`: stitched vertex saliency and curvature.
- `DSGS_50.json`: configuration, versions, file hashes, face counts and timing.

Saliency is used by surface fitting. The current decimation call does not use
vertex-quality weighting. Record the actual face count and configuration when
reporting results. Fixed seeds support repeated runs; numerical results may
still vary with hardware and dependency versions.

## License

The source code is licensed under [Apache License 2.0](LICENSE).
Third-party dependencies retain their respective licenses.
