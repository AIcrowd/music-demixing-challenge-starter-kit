from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from typing import List, Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lasaft.source_separation.conditioned import loss_functions
from lasaft.utils import fourier
from lasaft.utils.fourier import get_trim_length
from lasaft.utils.functions import get_optimizer_by_name, get_estimation
from lasaft.utils.weight_initialization import init_weights_functional


class Conditional_Source_Separation(pl.LightningModule, metaclass=ABCMeta):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--auto_lr_schedule', type=bool, default=False)

        return loss_functions.add_model_specific_args(parser)

    def __init__(self, n_fft, hop_length, num_frame, optimizer, lr, auto_lr_schedule):
        super(Conditional_Source_Separation, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.trim_length = get_trim_length(self.hop_length)
        self.n_trim_frames = self.trim_length // self.hop_length
        self.num_frame = num_frame

        self.lr = lr
        self.optimizer = optimizer
        self.auto_lr_schedule = auto_lr_schedule

        self.target_names = ['vocals', 'drums', 'bass', 'other']

    def configure_optimizers(self):
        optimizer = get_optimizer_by_name(self.optimizer)
        optimizer = optimizer(self.parameters(), lr=float(self.lr))

        if self.auto_lr_schedule:
            # pl does not support below when check_val_every_n_epoch != 1
            assert self.trainer.check_val_every_n_epoch == 1
            return {
                'optimizer': optimizer,
                'lr_scheduler': ReduceLROnPlateau(optimizer,
                                                  factor=0.1,
                                                  patience=50,
                                                  verbose=True,
                                                  ),
                'reduce_on_plateau': True,
                'interval': 'epoch',
                'monitor': 'val_loss'
            }
        else:
            return optimizer

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def forward(self, input_signal, input_condition) -> torch.Tensor:
        pass

    @abstractmethod
    def separate(self, input_signal, input_condition) -> torch.Tensor:
        pass

    @abstractmethod
    def init_weights(self):
        pass


class Spectrogram_based(Conditional_Source_Separation, metaclass=ABCMeta):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--n_fft', type=int, default=2048)
        parser.add_argument('--hop_length', type=int, default=1024)
        parser.add_argument('--num_frame', type=int, default=128)
        parser.add_argument('--spec_type', type=str, default='complex')
        parser.add_argument('--spec_est_mode', type=str, default='mapping')

        parser.add_argument('--train_loss', type=str, default='spec_mse')
        parser.add_argument('--val_loss', type=str, default='raw_l1')
        parser.add_argument('--unfreeze_stft_from', type=int, default=-1)  # -1 means never.

        return Conditional_Source_Separation.add_model_specific_args(parser)

    def __init__(self, n_fft, hop_length, num_frame,
                 spec_type, spec_est_mode,
                 conditional_spec2spec,
                 optimizer, lr, auto_lr_schedule,
                 train_loss, val_loss
                 ):
        super(Spectrogram_based, self).__init__(n_fft, hop_length, num_frame,
                                                optimizer, lr, auto_lr_schedule)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frame = num_frame

        assert spec_type in ['magnitude', 'complex']
        assert spec_est_mode in ['masking', 'mapping']
        self.magnitude_based = spec_type == 'magnitude'
        self.masking_based = spec_est_mode == 'masking'
        self.stft = fourier.multi_channeled_STFT(n_fft=n_fft, hop_length=hop_length)
        self.stft.freeze()

        self.spec2spec = conditional_spec2spec
        self.valid_estimation_dict = {}
        self.val_loss = val_loss
        self.train_loss = train_loss

        self.init_weights()

    def init_weights(self):
        init_weights_functional(self.spec2spec,
                                self.spec2spec.activation)

    def training_step(self, batch, batch_idx):
        mixture_signal, target_signal, condition = batch
        loss = self.train_loss(self, mixture_signal, condition, target_signal)
        self.log('train_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 reduce_fx=torch.mean)
        return loss

    # Validation Process
    def on_validation_epoch_start(self):
        self.num_val_item = len(self.val_dataloader().dataset)
        for target_name in self.target_names:
            self.valid_estimation_dict[target_name] = {mixture_idx: {}
                                                       for mixture_idx
                                                       in range(14)}

    def validation_step(self, batch, batch_idx):

        mixtures, targets, mixture_ids, window_offsets, input_conditions, target_names = batch
        batch_size = mixtures.shape[0]

        loss = self.val_loss(self, mixtures, input_conditions, targets) * batch_size / self.num_val_item

        self.log('raw_val_loss', loss, prog_bar=False, logger=False, reduce_fx=torch.sum)

        # Result Cache
        if 0 in mixture_ids.view(-1):
            estimated_targets = self.separate(mixtures, input_conditions)[:, self.trim_length:-self.trim_length]
            # targets = targets[:, self.trim_length:-self.trim_length]

            for mixture, mixture_idx, window_offset, input_condition, target_name, estimated_target \
                    in zip(mixtures, mixture_ids, window_offsets, input_conditions, target_names, estimated_targets):

                if mixture_idx == 0:
                    self.valid_estimation_dict[target_name][mixture_idx.item()][
                        window_offset.item()] = estimated_target.detach().cpu().numpy()
        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        if self.current_epoch % 20 == 0:
            for idx in [0]:
                estimation = {}
                for target_name in self.target_names:
                    estimation[target_name] = get_estimation(idx, target_name, self.valid_estimation_dict)
                    if estimation[target_name] is None:
                        continue
                    if estimation[target_name] is not None:
                        estimation[target_name] = estimation[target_name].astype(np.float32)

        reduced_loss = torch.stack(outputs).sum()
        self.log('val_loss', reduced_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 reduce_fx=torch.sum, sync_dist=True, sync_dist_op='sum')
        print(reduced_loss)

    def on_test_epoch_start(self):
        self.valid_estimation_dict = None
        self.test_estimation_dict = {}
        self.musdb_test = self.test_dataloader().dataset
        num_tracks = self.musdb_test.num_tracks
        for target_name in self.target_names:
            self.test_estimation_dict[target_name] = {mixture_idx: {}
                                                      for mixture_idx
                                                      in range(num_tracks)}

    @abstractmethod
    def to_spec(self, input_signal) -> torch.Tensor:
        pass

    @abstractmethod
    def separate(self, input_signal, input_condition) -> torch.Tensor:
        pass
