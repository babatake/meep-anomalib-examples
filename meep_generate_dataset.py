#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meep 2D dataset generator for anomalib
- Normal: uniform medium, no particle
- Anomaly: one circular particle
- Output: grayscale PNG of log10(|Ez|^2 + eps), normalized to 0..255

Run examples:
  conda activate meep-anomalib
  python meep_generate_dataset.py --outdir data --n_train 200 --n_test 40
"""

import argparse
from pathlib import Path
import numpy as np

import meep as mp
import imageio.v2 as imageio


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize float image to 0..255 uint8 safely."""
    mn = float(np.nanmin(img))
    mx = float(np.nanmax(img))
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-15:
        return np.zeros_like(img, dtype=np.uint8)
    x = (img - mn) / (mx - mn)
    x = np.clip(x, 0.0, 1.0)
    return (255.0 * x).astype(np.uint8)


def run_one(
    out_png: Path,
    with_particle: bool,
    *,
    seed: int,
    # Cell / resolution
    sx: float,
    sy: float,
    resolution: int,
    pml: float,
    # Source
    wavelength: float,
    fwidth: float,
    src_margin: float,
    src_shift_x: float,
    src_shift_y: float,
    # Particle
    particle_r: float,
    particle_n: float,
    particle_x: float,
    particle_y: float,
    # Field dump
    log_eps: float,
    # Run control
    until: float,
    decay_by: float,
    decay_at: float,
    decay_thresh: float,
    # Optional saves
    save_npy: bool,
):
    """
    Generate one sample image from Meep 2D TM (Ez).
    """
    rng = np.random.default_rng(seed)

    cell = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(pml)]

    geometry = []
    if with_particle:
        geometry.append(
            mp.Cylinder(
                radius=particle_r,
                height=mp.inf,
                center=mp.Vector3(particle_x, particle_y),
                material=mp.Medium(index=particle_n),
            )
        )

    # frequency in 1/um if lengths in um and c=1
    freq = 1.0 / wavelength

    # Source: a line source spanning y (approx plane wave), shifted a bit to make normal set diverse.
    src_x = -0.5 * sx + pml + src_margin + src_shift_x
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency=freq, fwidth=fwidth),
            component=mp.Ez,
            center=mp.Vector3(src_x, src_shift_y),
            size=mp.Vector3(0, sy - 2 * pml, 0),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        default_material=mp.Medium(index=1.0),
        resolution=resolution,
        dimensions=2,
    )

    # --- run ---
    # Option A: fixed until
    # Option B: stop when fields have decayed at a point (more robust)
    if decay_by > 0:
        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                decay_by, mp.Ez, mp.Vector3(decay_at, 0), decay_thresh
            ),
            until=until,
        )
    else:
        sim.run(until=until)

    # --- dump field to image ---
    ez = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sx, sy), component=mp.Ez)
    I = np.abs(ez) ** 2

    # Log scale improves visibility of weak scattering features
    Ilog = np.log10(I + log_eps)

    img_u8 = normalize_to_uint8(Ilog)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_png.as_posix(), img_u8)

    if save_npy:
        npy_path = out_png.with_suffix(".npy")
        np.save(npy_path.as_posix(), I.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="data", help="output root directory")
    ap.add_argument("--n_train", type=int, default=200, help="number of normal training images")
    ap.add_argument("--n_test", type=int, default=40, help="number of test images (mix normal/anom)")
    ap.add_argument("--anom_ratio", type=float, default=0.5, help="fraction of anomalies in test set")
    ap.add_argument("--save_npy", action="store_true", help="also save raw |Ez|^2 as .npy")

    # geometry / sim params
    ap.add_argument("--sx", type=float, default=16.0, help="cell size x (um)")
    ap.add_argument("--sy", type=float, default=10.0, help="cell size y (um)")
    ap.add_argument("--resolution", type=int, default=40, help="pixels per um")
    ap.add_argument("--pml", type=float, default=1.0, help="PML thickness (um)")

    # source params
    ap.add_argument("--wavelength", type=float, default=1.55, help="wavelength (um)")
    ap.add_argument("--fwidth", type=float, default=0.2, help="GaussianSource fwidth (in 1/um)")
    ap.add_argument("--src_margin", type=float, default=1.0, help="source margin from PML (um)")
    ap.add_argument("--src_jitter_x", type=float, default=0.2, help="source x jitter range (um)")
    ap.add_argument("--src_jitter_y", type=float, default=0.3, help="source y jitter range (um)")

    # particle params
    ap.add_argument("--particle_r", type=float, default=0.30, help="particle radius (um)")
    ap.add_argument("--particle_n", type=float, default=2.0, help="particle refractive index")
    ap.add_argument("--particle_jitter", type=float, default=0.2, help="particle position jitter (um)")

    # run control
    ap.add_argument("--until", type=float, default=200.0, help="max simulation time")
    ap.add_argument("--decay_by", type=float, default=50.0,
                    help="use stop_when_fields_decayed with this time after sources (0 disables)")
    ap.add_argument("--decay_at", type=float, default=6.0,
                    help="x-position (um) for decay monitor point")
    ap.add_argument("--decay_thresh", type=float, default=1e-6, help="decay threshold")

    # dump params
    ap.add_argument("--log_eps", type=float, default=1e-12, help="epsilon inside log10")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    train_dir = outdir / "train" / "good"
    test_good_dir = outdir / "test" / "good"
    test_anom_dir = outdir / "test" / "anomaly"

    # --- generate training normals ---
    for i in range(args.n_train):
        seed = i
        # jitter source a bit to create a "distribution" of normal conditions
        sxj = np.random.uniform(-args.src_jitter_x, args.src_jitter_x)
        syj = np.random.uniform(-args.src_jitter_y, args.src_jitter_y)

        out_png = train_dir / f"normal_{i:04d}.png"
        run_one(
            out_png,
            with_particle=False,
            seed=seed,
            sx=args.sx, sy=args.sy, resolution=args.resolution, pml=args.pml,
            wavelength=args.wavelength, fwidth=args.fwidth,
            src_margin=args.src_margin, src_shift_x=sxj, src_shift_y=syj,
            particle_r=args.particle_r, particle_n=args.particle_n,
            particle_x=0.0, particle_y=0.0,
            log_eps=args.log_eps,
            until=args.until,
            decay_by=args.decay_by,
            decay_at=args.decay_at,
            decay_thresh=args.decay_thresh,
            save_npy=args.save_npy,
        )

    # --- generate test set (mix normal/anom) ---
    n_anom = int(round(args.n_test * args.anom_ratio))
    n_good = args.n_test - n_anom

    # test good
    for i in range(n_good):
        seed = 10_000 + i
        sxj = np.random.uniform(-args.src_jitter_x, args.src.src_jitter_x) if False else np.random.uniform(-args.src_jitter_x, args.src_jitter_x)
        syj = np.random.uniform(-args.src_jitter_y, args.src_jitter_y)
        out_png = test_good_dir / f"good_{i:04d}.png"
        run_one(
            out_png,
            with_particle=False,
            seed=seed,
            sx=args.sx, sy=args.sy, resolution=args.resolution, pml=args.pml,
            wavelength=args.wavelength, fwidth=args.fwidth,
            src_margin=args.src_margin, src_shift_x=sxj, src_shift_y=syj,
            particle_r=args.particle_r, particle_n=args.particle_n,
            particle_x=0.0, particle_y=0.0,
            log_eps=args.log_eps,
            until=args.until,
            decay_by=args.decay_by,
            decay_at=args.decay_at,
            decay_thresh=args.decay_thresh,
            save_npy=args.save_npy,
        )

    # test anomaly
    for i in range(n_anom):
        seed = 20_000 + i
        sxj = np.random.uniform(-args.src_jitter_x, args.src_jitter_x)
        syj = np.random.uniform(-args.src_jitter_y, args.src_jitter_y)

        # jitter particle position (small)
        px = np.random.uniform(-args.particle_jitter, args.particle_jitter)
        py = np.random.uniform(-args.particle_jitter, args.particle_jitter)

        out_png = test_anom_dir / f"anom_{i:04d}.png"
        run_one(
            out_png,
            with_particle=True,
            seed=seed,
            sx=args.sx, sy=args.sy, resolution=args.resolution, pml=args.pml,
            wavelength=args.wavelength, fwidth=args.fwidth,
            src_margin=args.src_margin, src_shift_x=sxj, src_shift_y=syj,
            particle_r=args.particle_r, particle_n=args.particle_n,
            particle_x=px, particle_y=py,
            log_eps=args.log_eps,
            until=args.until,
            decay_by=args.decay_by,
            decay_at=args.decay_at,
            decay_thresh=args.decay_thresh,
            save_npy=args.save_npy,
        )

    print(f"Done. Dataset written to: {outdir.resolve()}")
    print(f"Train normals: {train_dir}")
    print(f"Test good    : {test_good_dir}")
    print(f"Test anomaly : {test_anom_dir}")


if __name__ == "__main__":
    main()
