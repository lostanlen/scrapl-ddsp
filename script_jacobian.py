"""
This script computes the Jacobian of (Phi \circ g)
for one random seed of the chirp texture dataset.
"""
import fire
from kymatio.torch import TimeFrequencyScattering
import os
import torch

import cnn
import synth

def run(density_idx, slope_idx, seed, sav_dir):
    dataset = cnn.ChirpTextureDataModule(
        n_densities=7,
        n_slopes=7,
        n_folds=1,
        batch_size=1)
    dataset.setup()

    sub_df = dataset.df[
        (dataset.df['density_idx']==density_idx) &
        (dataset.df['slope_idx']==slope_idx)
    ]
    if len(sub_df) == 1:
        row = sub_df.iloc[0]
        theta = torch.tensor([row["density"], row["slope"]], requires_grad=True)
    else:
        raise ValueError("Expected one row, got:\n {}".format(sub_df))
    
    sc = TimeFrequencyScattering(
        shape=(2**15),
        J=6,
        Q=(24, 2),
        Q_fr=2,
        J_fr=5,
        T='global',
        F='global',
        format='time',
    )

    def S_from_theta(theta, sc, dataset, seed):
        x = synth.generate_chirp_texture(
            theta_density=theta[0],
            theta_slope=theta[1],
            duration=dataset.train_ds.duration,
            event_duration=dataset.train_ds.event_duration,
            sr=dataset.train_ds.sr,
            fmin=dataset.train_ds.fmin,
            fmax=dataset.train_ds.fmax,
            n_events=dataset.train_ds.n_events,
            Q=dataset.train_ds.Q,
            hop_length=dataset.train_ds.hop_length,
            seed=seed,
        )
        return sc(x)

    S_closure = lambda theta: S_from_theta(theta, sc, dataset, seed)
    J = torch.autograd.functional.jacobian(S_closure, theta)

    # save the Jacobian to sav_dir with formatted name
    sav_name = "jacobian_density-{}_slope-{}_seed-{}.pt".format(
        density_idx, slope_idx, seed)
    jac_dir = os.path.join(sav_dir, "jacobians_7x7")
    os.makedirs(jac_dir, exist_ok=True)
    sav_path = os.path.join(jac_dir, sav_name)
    torch.save(J.squeeze(), sav_path)


if __name__ == '__main__':
    fire.Fire(run)
