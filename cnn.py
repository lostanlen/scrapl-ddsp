import auraloss
import itertools
import loss
import metrics
from nnAudio.features import CQT
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

import synth


class EffNet(pl.LightningModule):
    def __init__(self, loss_type, outdim, save_path, steps_per_epoch):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm2d(
            1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        # adapt to EfficientNet's mandatory 3 input channels
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))
        self.model = torchvision.models.efficientnet_b0(num_classes=outdim)
        self.n_batches_train = steps_per_epoch

        self.loss_type = loss_type
        if self.loss_type == "ploss":
            self.loss = F.mse_loss
        elif self.loss_type == "mss":
            self.loss = loss.loss_spec
            self.specloss = auraloss.freq.MultiResolutionSTFTLoss()
        elif self.loss_type == "jtfs":
            self.loss = loss.TimeFrequencyScatteringLoss()
        # TODO: VGGish loss

        self.save_path = save_path

        self.val_loss = None
        self.outdim = outdim

        self.metric_macro = metrics.JTFSloss(self.scaler)
        self.metric_mss = metrics.MSSloss(self.scaler)

        self.monitor_valloss = torch.inf
        self.current_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.best_params = self.parameters
        self.epoch = 0

        self.test_preds = []
        self.test_gts = []

        self.train_outputs = []
        self.test_outputs = []
        self.val_outputs = []

    def forward(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        x = self.batchnorm1(input_tensor)
        x = self.conv2d(x)
        x = self.model(x)
        return x


class ChirpTextureData(Dataset):
    def __init__(self, df):
        super().__init__()

        self.df = df
        self.J = 6
        self.Q = 24
        self.sr = 2**13
        self.hop_length = 2**6

        self.cqt_epsilon = 1e-3
        self.duration = 4
        self.event_duration = 2**(-4)
        self.fmin = 2**8
        self.fmax = 2**11
        self.n_events = 2**6
        self.sr = 2**13
     
        # define CQT closure
        cqt_params = {
            'sr': self.sr,
            'bins_per_octave': self.Q,
            'n_bins': self.J * self.Q,
            'hop_length': self.hop_length,
            'fmin': (0.4*self.sr) / (2**self.J)
        }   
        if torch.cuda.is_available():
            self.cqt_from_x = CQT(**cqt_params).cuda()
        else:
            self.cqt_from_x = CQT(**cqt_params)

    def __getitem__(self, idx):
        theta_density = self.df.iloc[idx]["density"]
        theta_slope = self.df.iloc[idx]["slope"]

        x = synth.generate_chirp_texture(
            theta_density,
            theta_slope,
            duration=self.duration,
            event_duration=self.event_duration,
            sr=self.sr,
            fmin=self.fmin,
            fmax=self.fmax,
            n_events=self.n_events,
            Q=self.Q,
            hop_length=self.hop_length,
        )
        U = self.cqt_from_x(x)
        return {'feature': U, 'density': theta_density, 'slope': theta_slope}

    def cqt_from_x(self, x):
        CQT_x = self.cqt_from_x(x).abs()
        return torch.log1p(CQT_x / self.cqt_epsilon)
        

class ChirpTextureDataModule(pl.LightningDataModule):
    def __init__(self, *, n_densities, n_slopes, n_folds, batch_size):
        slopes = torch.linspace(-1, 1, n_slopes + 2)[1:-1]
        densities = torch.linspace(0, 1, n_densities + 2)[1:-1]

        thetas = list(itertools.product(slopes, densities))
        df = pd.DataFrame(thetas, columns=["slope", "density"])

        folds = torch.linspace(0, len(df), n_folds + 1).int()
        shuffling_idx = np.random.RandomState(seed=42).permutation(10)
        df["fold"] = folds[shuffling_idx]

        train_df = df[df["fold"] < (n_folds - 2)]
        self.train_ds = ChirpTextureData(train_df)

        val_df = df[df["fold"] == (n_folds - 2)]
        self.val_ds = ChirpTextureData(val_df)
        
        test_df = df[df["fold"] == (n_folds - 1)]
        self.test_ds = ChirpTextureData(test_df)

        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
