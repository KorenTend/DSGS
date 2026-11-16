import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from utils_geo import compute_curvature, get_patches, stitch_weights
from model import DentalSaliencyNet
from simplify import WeightedQEMSimplifier
from runtime import (setup_seed, load_mesh, positive_int, retention_ratio,
                     resolve_device, sha256_file)
from loss import (virtual_decimation_loss, view_consistency_loss, sparsity_loss,
                  edge_preserving_loss, chamfer_distance_charbonnier)


def create_model(device, ablation):
    model = DentalSaliencyNet().to(device)
    if ablation == 'no_gam':
        model.encoder.attention = torch.nn.Identity()
    if ablation == 'no_learn_curv':
        with torch.no_grad():
            model.encoder.blend_alpha.fill_(0.8473)
        model.encoder.blend_alpha.requires_grad_(False)
    return model


def prepare_patches(mesh, cfg):
    curvature = compute_curvature(mesh)
    patches = get_patches(mesh, curvature, cfg)
    if not patches:
        raise ValueError('No patches contain 100 vertices; use a denser mesh or fewer patches')
    return curvature, patches


def train(model, patches, cfg, device, ablation):
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    model.train()

    for epoch in range(cfg.EPOCHS):
        total_loss = 0
        for patch in patches:

            pts = torch.from_numpy(patch['points_norm']).float().to(device)
            nms = torch.from_numpy(patch['normals']).float().to(device)
            curv = torch.from_numpy(patch['curvature']).float().to(device).unsqueeze(1)
            height = torch.from_numpy(patch['height']).float().to(device).unsqueeze(1)


            x = torch.cat([pts, nms, curv, height], dim=1).unsqueeze(0)
            curv_input = curv.unsqueeze(0)


            theta = np.random.uniform(0, 2*np.pi)
            rot_mat = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]]).float().to(device)
            pts_rot = torch.matmul(pts, rot_mat.T)
            nms_rot = torch.matmul(nms, rot_mat.T)
            x_rot = torch.cat([pts_rot, nms_rot, curv, height], dim=1).unsqueeze(0)

            optimizer.zero_grad()
            weights, recon_pts, weights_rot = model(x, curv_input, x_rot, ablate=ablation)


            if ablation == 'no_virtual_loss':
                loss_geom = torch.tensor(0.0, device=device)
            else:
                loss_geom = virtual_decimation_loss(pts.unsqueeze(0), weights, curv.unsqueeze(0))


            loss_sparse = sparsity_loss(weights)


            loss_view = view_consistency_loss(weights, weights_rot)


            loss_recon = chamfer_distance_charbonnier(x[:, :, :3], recon_pts)


            loss_edge = edge_preserving_loss(weights, curv_input)


            loss = 1.0 * loss_geom + 2.0 * loss_sparse + 0.5 * loss_view + 1.0 * loss_recon + 0.5 * loss_edge

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch+1) % 10 == 0 or epoch + 1 == cfg.EPOCHS:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{cfg.EPOCHS}, Avg Loss: {total_loss/len(patches):.4f}, LR: {current_lr:.6f}")


def infer_weights(model, mesh, cfg, device, ablation):
    curvature, patches = prepare_patches(mesh, cfg)
    model.eval()
    patch_weights = []
    with torch.no_grad():
        for patch in patches:
            features = np.column_stack((patch['points_norm'], patch['normals'],
                                        patch['curvature'], patch['height']))
            x = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
            weights, _, _ = model(x, x[:, :, 6:7], ablate=ablation)
            patch_weights.append(weights.cpu().numpy())
    weights = stitch_weights(mesh, patches, patch_weights)
    weights = (weights - weights.min()) / (np.ptp(weights) + 1e-8)
    return weights, curvature


def build_parser():
    parser = argparse.ArgumentParser(description='Train DSGS on a mesh, or simplify with a saved saliency checkpoint.')
    parser.add_argument('--input', '-i', type=Path, default=Path('data/Origin-10.stl'))
    parser.add_argument('--output', '-o', type=Path, default=Path('output/DSGS.obj'))
    parser.add_argument('--ratio', '--target-ratio', '-r', type=retention_ratio,
                        default=Config.TARGET_RATIO, help='fraction of faces retained, in (0, 1]')
    parser.add_argument('--device', '-d', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=positive_int, default=Config.EPOCHS)
    parser.add_argument('--seed', type=int, default=Config.SEED)
    parser.add_argument('--patch-points', type=positive_int, default=Config.PATCH_POINTS)
    parser.add_argument('--num-patches', type=positive_int, default=Config.NUM_PATCHES)
    parser.add_argument('--checkpoint', type=Path, help='skip training and load saved weights')
    parser.add_argument('--no-surface-refinement', action='store_true')
    parser.add_argument('--ablate', default='',
                        choices=['', 'no_virtual_loss', 'no_gam', 'no_learn_curv'])
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f'input mesh not found: {args.input}; see readme.md')
    if args.patch_points < 20:
        parser.error('--patch-points must be at least 20 for the network KNN')
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 2**32 - 1')
    if args.output.suffix.lower() not in {'.obj', '.ply', '.stl'}:
        parser.error('output must use .obj, .ply or .stl')
    if args.input.resolve() == args.output.resolve():
        parser.error('input and output must be different files')
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f'checkpoint not found: {args.checkpoint}')
    if args.checkpoint and args.output.with_suffix('.pth').resolve() == args.checkpoint.resolve():
        parser.error('choose a new output stem to preserve the training run metadata')
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    cfg = Config()
    cfg.SEED, cfg.EPOCHS = args.seed, args.epochs
    cfg.PATCH_POINTS, cfg.NUM_PATCHES = args.patch_points, args.num_patches
    setup_seed(cfg.SEED)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            if args.ablate and args.ablate != checkpoint['ablation']:
                parser.error('--ablate does not match the checkpoint variant')
            args.ablate = checkpoint['ablation']
            for key, value in checkpoint['config'].items():
                setattr(cfg, key, value)
    model = create_model(device, args.ablate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_seconds = 0.0
    config = {key: getattr(cfg, key) for key in dir(cfg) if key.isupper()}
    if checkpoint is not None:
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        checkpoint_path = args.checkpoint
    else:
        start = time.perf_counter()
        mesh = load_mesh(args.input)
        _, patches = prepare_patches(mesh, cfg)
        print(f'Training {args.ablate or "full"} on {device}; {len(patches)} patches', flush=True)
        train(model, patches, cfg, device, args.ablate)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - start
        checkpoint_path = args.output.with_suffix('.pth')
        torch.save({'format_version': 1, 'model_state_dict': model.state_dict(),
                    'ablation': args.ablate, 'config': config,
                    'training_input_sha256': sha256_file(args.input)}, checkpoint_path)


    setup_seed(cfg.SEED)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    reference = load_mesh(args.input, clean=False) if not args.no_surface_refinement else None
    mesh = load_mesh(args.input)
    weights, curvature = infer_weights(model, mesh, cfg, device, args.ablate)
    weights_path = args.output.with_name(args.output.stem + '_weights.npz')
    np.savez(weights_path, weights=weights, curvature=curvature)
    target = min(len(mesh.faces), max(500, int(len(mesh.faces) * args.ratio)))
    result = WeightedQEMSimplifier(mesh, weights.copy()).simplify(target)
    if reference is not None:
        from surface_refinement import refine_surface
        result = refine_surface(reference, result, mesh.vertices, weights)
    if reference is not None and args.output.suffix.lower() == '.obj':
        result.export(args.output, include_normals=False)
    else:
        result.export(args.output)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    pipeline_seconds = time.perf_counter() - start
    metadata = dict(method='DSGS', ablation=args.ablate or 'full',
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    input=str(args.input), input_sha256=sha256_file(args.input),
                    output=str(args.output), output_sha256=sha256_file(args.output),
                    checkpoint=str(checkpoint_path), checkpoint_sha256=sha256_file(checkpoint_path),
                    trained_this_run=checkpoint is None, config=config, device=str(device),
                    surface_refinement=reference is not None, qem_quality_weight=False,
                    requested_ratio=args.ratio, target_faces=target,
                    original_faces=len(mesh.faces), output_faces=len(result.faces),
                    actual_ratio=len(result.faces) / len(mesh.faces),
                    training_seconds=training_seconds, pipeline_seconds=pipeline_seconds,
                    versions={p: importlib.metadata.version(p) for p in
                              ['torch', 'numpy', 'scipy', 'trimesh', 'pymeshlab', 'open3d']})
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Done. Output saved to: {args.output}')
    print(f'[TIME] Training and preprocessing: {training_seconds:.3f}s')
    print(f'[TIME] Full Pipeline (Load+Inference+Simplify+Export): {pipeline_seconds:.3f}s')


if __name__ == '__main__':
    main()
