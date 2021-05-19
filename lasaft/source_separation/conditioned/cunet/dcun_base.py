from argparse import ArgumentParser
from typing import Tuple
from warnings import warn

import torch
import torch.nn as nn
from torch import Tensor

from lasaft.data.musdb_wrapper import SingleTrackSet
from lasaft.source_separation.conditioned.separation_framework import Spectrogram_based
from lasaft.utils.functions import get_activation_by_name, string_to_list


class Dense_CUNet(nn.Module):

    def __init__(self,
                 n_fft,
                 input_channels, internal_channels,
                 n_blocks, n_internal_layers,
                 mk_block_f, mk_ds_f, mk_us_f,
                 first_conv_activation, last_activation,
                 t_down_layers, f_down_layers,

                 # Conditional Mechanism #
                 control_vector_type, control_input_dim, embedding_dim, condition_to
                 ):

        first_conv_activation = get_activation_by_name(first_conv_activation)
        last_activation = get_activation_by_name(last_activation)

        super(Dense_CUNet, self).__init__()

        '''num_block should be an odd integer'''
        assert n_blocks % 2 == 1

        dim_f, t_down_layers, f_down_layers = self.mk_overall_structure(n_fft, internal_channels, input_channels,
                                                                        n_blocks,
                                                                        n_internal_layers, last_activation,
                                                                        first_conv_activation,
                                                                        t_down_layers, f_down_layers)

        self.mk_blocks(dim_f, internal_channels, mk_block_f, mk_ds_f, mk_us_f, t_down_layers)

        #########################
        # Conditional Mechanism #
        #########################
        assert control_vector_type in ['one_hot_mode', 'embedding']
        if control_vector_type == 'one_hot_mode':
            if control_input_dim != embedding_dim:
                warn('in one_hot_mode, embedding_dim should be the same as num_targets. auto correction')
                embedding_dim = control_input_dim

                with torch.no_grad():
                    one_hot_weight = torch.zeros((control_input_dim, embedding_dim))
                    for i in range(control_input_dim):
                        one_hot_weight[i, i] = 1.

                    self.embedding = nn.Embedding(control_input_dim, embedding_dim, _weight=one_hot_weight)
                    self.embedding.weight.requires_grad = True
        elif control_vector_type == 'embedding':
            self.embedding = nn.Embedding(control_input_dim, embedding_dim)

        # Where to condition
        assert condition_to in ['encoder', 'decoder', 'full']
        self.is_encoder_conditioned = self.is_middle_conditioned = self.is_decoder_conditioned = False
        if condition_to == 'encoder':
            self.is_encoder_conditioned = True
        elif condition_to == 'decoder':
            self.is_decoder_conditioned = True
        elif condition_to == 'full':
            self.is_encoder_conditioned = self.is_middle_conditioned = self.is_decoder_conditioned = True
        else:
            raise NotImplementedError

        self.activation = self.last_conv[-1]

    def mk_blocks(self, dim_f, internal_channels, mk_block_f, mk_ds_f, mk_us_f, t_down_layers):
        f = dim_f
        for i in range(self.n):
            self.encoders.append(mk_block_f(internal_channels, internal_channels, f))
            ds_layer, f = mk_ds_f(internal_channels, i, f, t_down_layers)
            self.downsamplings.append(ds_layer)
        self.mid_block = mk_block_f(internal_channels, internal_channels, f)
        for i in range(self.n):
            us_layer, f = mk_us_f(internal_channels, i, f, self.n, t_down_layers)
            self.upsamplings.append(us_layer)
            self.decoders.append(mk_block_f(2 * internal_channels, internal_channels, f))

    def mk_overall_structure(self, n_fft, internal_channels, input_channels, n_blocks, n_internal_layers,
                             last_activation, first_conv_activation, t_down_layers, f_down_layers):
        dim_f = n_fft // 2
        input_channels = input_channels
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=internal_channels,
                kernel_size=(1, 2),
                stride=1
            ),
            nn.BatchNorm2d(internal_channels),
            first_conv_activation(),
        )
        self.encoders = nn.ModuleList()
        self.downsamplings = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamplings = nn.ModuleList()
        self.last_conv = nn.Sequential(

            nn.Conv2d(
                in_channels=internal_channels,
                out_channels=input_channels,
                kernel_size=(1, 2),
                stride=1,
                padding=(0, 1)
            ),
            last_activation()
        )
        self.n = n_blocks // 2
        if t_down_layers is None:
            t_down_layers = list(range(self.n))
        elif n_internal_layers == 'None':
            t_down_layers = list(range(self.n))
        else:
            t_down_layers = string_to_list(t_down_layers)
        if f_down_layers is None:
            f_down_layers = list(range(self.n))
        elif n_internal_layers == 'None':
            f_down_layers = list(range(self.n))
        else:
            f_down_layers = string_to_list(f_down_layers)
        return dim_f, t_down_layers, f_down_layers

    def forward(self, input_spec, input_condition):

        condition_embedding = self.embedding(input_condition)
        gammas, betas = self.condition_generator(condition_embedding)

        x = self.first_conv(input_spec)
        encoding_outputs = []

        for encoder, downsampling, gamma, beta in zip(self.encoders, self.downsamplings, gammas, betas):
            x = encoder(x)
            x = self.film(x, gamma, beta)
            encoding_outputs.append(x)
            x = downsampling(x)

        x = self.mid_block(x)

        for i in range(self.n):
            x = self.upsamplings[i](x)
            x = torch.cat((x, encoding_outputs[-i - 1]), 1)
            x = self.decoders[i](x)

        return self.last_conv(x)


class Dense_CUNet_Framework(Spectrogram_based):

    def __init__(self, n_fft, hop_length, num_frame,
                 spec_type, spec_est_mode,
                 spec2spec,
                 optimizer, lr, auto_lr_schedule,
                 train_loss, val_loss
                 ):

        super(Dense_CUNet_Framework, self).__init__(
            n_fft, hop_length, num_frame,
            spec_type, spec_est_mode,
            spec2spec,
            optimizer, lr, auto_lr_schedule,
            train_loss, val_loss
        )

    def to_spec(self, input_signal) -> torch.Tensor:
        if self.magnitude_based:
            return self.stft.to_mag(input_signal).transpose(-1, -3)
        else:
            spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
            spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
            return spec_complex.transpose(-1, -3)  # *, 2ch, T, N

    def forward(self, input_signal, input_condition) -> torch.Tensor:
        input_spec = self.to_spec(input_signal)
        output_spec = self.spec2spec(input_spec, input_condition)

        if self.masking_based:
            output_spec = input_spec * output_spec

        return output_spec

    def separate(self, input_signal, input_condition) -> torch.Tensor:

        phase = None
        if self.magnitude_based:
            mag, phase = self.stft.to_mag_phase(input_signal)
            input_spec = mag.transpose(-1, -3)

        else:
            spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
            spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
            input_spec = spec_complex.transpose(-1, -3)  # *, 2ch, T, N

        output_spec = self.spec2spec(input_spec, input_condition)

        if self.masking_based:
            output_spec = input_spec * output_spec
        else:
            pass  # Use the original output_spec

        output_spec = output_spec.transpose(-1, -3)

        if self.magnitude_based:
            restored = self.stft.restore_mag_phase(output_spec, phase)
        else:
            # output_spec: *, N, T, 2ch
            output_spec = output_spec.view(list(output_spec.shape)[:-1] + [2, -1])  # *, N, T, 2, ch
            restored = self.stft.restore_complex(output_spec)

        return restored

    def separate_and_return_spec(self, input_signal, input_condition) -> Tuple[Tensor, Tensor]:

        phase = None
        if self.magnitude_based:
            mag, phase = self.stft.to_mag_phase(input_signal)
            input_spec = mag.transpose(-1, -3)

        else:
            spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
            spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
            input_spec = spec_complex.transpose(-1, -3)  # *, 2ch, T, N

        output_spec = self.spec2spec(input_spec, input_condition)

        if self.masking_based:
            output_spec = input_spec * output_spec
        else:
            pass  # Use the original output_spec

        output_spec_cache = output_spec
        output_spec = output_spec.transpose(-1, -3)

        if self.magnitude_based:
            restored = self.stft.restore_mag_phase(output_spec, phase)
        else:
            # output_spec: *, N, T, 2ch
            output_spec = output_spec.view(list(output_spec.shape)[:-1] + [2, -1])  # *, N, T, 2, ch
            restored = self.stft.restore_complex(output_spec)

        return restored, output_spec_cache

    def separate_track(self, input_signal, target, sf_mode=False) -> torch.Tensor:

        import numpy as np

        self.eval()
        with torch.no_grad():
                db = SingleTrackSet(input_signal, self.hop_length, self.num_frame)
                assert target in db.source_names
                separated = []

                input_condition = np.array(db.source_names.index(target))
                input_condition = torch.tensor(input_condition, dtype=torch.long, device=self.device).view(1)

                for item in db:
                    separated.append(self.separate(item.unsqueeze(0).to(self.device), input_condition)[0]
                                     [self.trim_length:-self.trim_length].detach().cpu().numpy())

        separated = np.concatenate(separated, axis=0)[:input_signal.shape[0]]

        if sf_mode:
            import soundfile
            soundfile.write('temp.wav', separated, 44100)
            return soundfile.read('temp.wav')[0]
        else:
            return separated

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--n_blocks', type=int, default=7)
        parser.add_argument('--input_channels', type=int, default=4)
        parser.add_argument('--internal_channels', type=int, default=24)
        parser.add_argument('--first_conv_activation', type=str, default='relu')
        parser.add_argument('--last_activation', type=str, default='identity')

        parser.add_argument('--t_down_layers', type=tuple, default=None)
        parser.add_argument('--f_down_layers', type=tuple, default=None)

        parser.add_argument('--control_vector_type', type=str, default='embedding')
        parser.add_argument('--control_input_dim', type=int, default=4)
        parser.add_argument('--embedding_dim', type=int, default=32)
        parser.add_argument('--condition_to', type=str, default='decoder')

        return Spectrogram_based.add_model_specific_args(parser)
