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
    def __init__(self, loss_type, save_path, steps_per_epoch):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm2d(
            1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        # adapt to EfficientNet's mandatory 3 input channels
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))
        self.effnet = torchvision.models.efficientnet_b0(num_classes=2)
        self.n_batches_train = steps_per_epoch

        self.loss_type = loss_type
        # if self.loss_type == "ploss":
        #     self.loss = F.mse_loss
        # elif self.loss_type == "mss":
        #     self.loss = loss.loss_spec
        #     self.specloss = auraloss.freq.MultiResolutionSTFTLoss()
        # elif self.loss_type == "jtfs":
        #     self.loss = loss.TimeFrequencyScatteringLoss()
        # TODO: VGGish loss

        self.save_path = save_path
        self.val_loss = None
        #self.metric_macro = metrics.JTFSloss()
        #self.metric_mss = metrics.MSSloss()
        self.monitor_valloss = torch.inf
        self.current_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_params = self.parameters
        self.epoch = 0

        self.test_preds = []
        self.test_gts = []
        self.train_outputs = []
        self.test_outputs = []
        self.val_outputs = []

        self.optimizer = self.configure_optimizers()

    def forward(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        x = self.batchnorm1(input_tensor)
        x = self.conv2d(x)
        x = self.effnet(x)
        density = torch.sigmoid(x[:, 0])
        slope = torch.tanh(x[:, 1])
        return {"density": density, "slope": slope}
    
    def step(self, batch, subset, batch_idx):
        U = batch["feature"].to(self.current_device)
        density = batch["density"].to(self.current_device)
        slope = batch["slope"].to(self.current_device)

        theta_pred = self(U)
        density_pred = theta_pred["density"]
        slope_pred = theta_pred["slope"]
        if self.loss_type == "ploss":
            density_loss = F.mse_loss(density_pred, density)
            slope_loss = F.mse_loss(slope_pred, slope)
            loss = density_loss + slope_loss

        if subset == 'train':
            self.train_outputs.append(loss)
            self.log("train_loss", loss, prog_bar=True)
        elif subset == 'test':
            self.test_outputs.append(loss)
            self.test_preds.append(theta_pred)
            theta_ref = {'density': density, 'slope': slope}
            self.test_gts.append(theta_ref)
        elif subset == 'val':
            self.val_outputs.append(loss)
        return {"loss": loss}
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train', batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val', batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test', batch_idx)
    
    def on_train_epoch_start(self):
        self.train_outputs = []
        self.test_outputs = []
        self.val_outputs = []
        self.log("lr", self.optimizer.param_groups[-1]['lr'])

    def on_train_epoch_end(self):
        avg_loss = torch.tensor(self.train_outputs).mean()
        self.log('train_loss', avg_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

class ChirpTextureData(Dataset):
    def __init__(self, df, seed):
        super().__init__()

        self.df = df
        self.seed = seed

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
            seed=self.seed,
        )
        U = self.cqt_from_x(x)
        return {'feature': U, 'density': theta_density, 'slope': theta_slope}
    
    def __len__(self):
        return len(self.df)

    def cqt_from_x(self, x):
        CQT_x = self.cqt_from_x(x).abs()
        return torch.log1p(CQT_x / self.cqt_epsilon)
        

class ChirpTextureDataModule(pl.LightningDataModule):
    def __init__(self, *, n_densities, n_slopes, n_folds, batch_size):
        super().__init__() 

        self.n_densities = n_densities
        self.n_slopes = n_slopes
        self.n_folds = n_folds
        self.batch_size = batch_size

        slopes = torch.linspace(-1, 1, n_slopes + 2)[1:-1]
        densities = torch.linspace(0, 1, n_densities + 2)[1:-1]

        thetas = list(itertools.product(densities, slopes))
        df = pd.DataFrame(thetas, columns=["density", "slope"])
        folds = torch.linspace(0, n_folds, len(df)).int()
        n_thetas = len(thetas)
        random_state = np.random.RandomState(seed=42)
        shuffling_idx = random_state.permutation(n_thetas)
        df["fold"] = folds[shuffling_idx]
        self.df = df

    def setup(self, stage=None):

        train_df = self.df[self.df["fold"] < (self.n_folds - 2)]
        self.train_ds = ChirpTextureData(train_df, seed=None)

        val_df = self.df[self.df["fold"] == (self.n_folds - 2)]
        self.val_ds = ChirpTextureData(val_df, seed=42)
        
        test_df = self.df[self.df["fold"] > (self.n_folds - 2)]
        self.test_ds = ChirpTextureData(test_df, seed=42)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
