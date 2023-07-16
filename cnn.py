import auraloss
import loss
import metrics
import pytorch_lightning as pl
import torch
from torch import nn
from torch import functional as F
import torchvision


class EffNet(pl.LightningModule):
    def __init__(self, loss_type, outdim, save_path, steps_per_epoch):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True)
        
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