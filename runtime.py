import argparse
import hashlib
import random

import numpy as np
import torch
import trimesh


def retention_ratio(value):
    ratio = float(value)
    if not 0 < ratio <= 1:
        raise argparse.ArgumentTypeError('face retention ratio must be in (0, 1]')
    return ratio


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu or --device auto')
    return torch.device(name)


def load_mesh(path, clean=True):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f'Expected a nonempty triangle mesh: {path}')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f'Mesh contains non-finite coordinates: {path}')
    if clean:
        mesh.process(validate=True)
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
    if not len(mesh.faces) or mesh.area <= 0:
        raise ValueError(f'Mesh has no valid surface after cleaning: {path}')
    return mesh


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
