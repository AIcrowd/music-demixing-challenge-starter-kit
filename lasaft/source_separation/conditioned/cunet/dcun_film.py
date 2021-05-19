from argparse import ArgumentParser
from typing import Tuple

import torch
from torch import Tensor

from lasaft.source_separation.conditioned.cunet.dcun_base import Dense_CUNet, Dense_CUNet_Framework
from lasaft.source_separation.sub_modules.control_models import dense_control_block, film_control_model
from lasaft.utils.FiLM_utils import FiLM_simple, FiLM_complex
from lasaft.utils.functions import get_activation_by_name


class DenseCUNet_FiLM(Dense_CUNet):

    def __init__(self,
                 n_fft,
                 input_channels, internal_channels,
                 n_blocks, n_internal_layers,
                 mk_block_f, mk_ds_f, mk_us_f,
                 first_conv_activation, last_activation,
                 t_down_layers, f_down_layers,
                 # Conditional Mechanism #
                 control_vector_type, control_input_dim, embedding_dim, condition_to,
                 control_type, control_n_layer, film_type, gamma_activation, beta_activation,
                 ):

        super(DenseCUNet_FiLM, self).__init__(
            n_fft,
            input_channels, internal_channels,
            n_blocks, n_internal_layers,
            mk_block_f, mk_ds_f, mk_us_f,
            first_conv_activation, last_activation,
            t_down_layers, f_down_layers,
            # Conditional Mechanism #
            control_vector_type, control_input_dim, embedding_dim, condition_to
        )

        assert film_type in ['simple', 'complex']
        self.film = FiLM_simple if film_type == 'simple' else FiLM_complex

        gamma_activation = get_activation_by_name(gamma_activation)
        beta_activation = get_activation_by_name(beta_activation)

        if control_type == "dense":
            control_block = dense_control_block(embedding_dim, control_n_layer)
            self.condition_generator = film_control_model(
                control_block, n_blocks, internal_channels,
                film_type,
                gamma_activation, beta_activation,
                condition_to
            )
        else:
            raise NotImplementedError

        self.activation = self.last_conv[-1]

    def forward(self, input_spec, input_condition):

        condition_embedding = self.embedding(input_condition)
        gammas, betas = self.condition_generator(condition_embedding)

        x = self.first_conv(input_spec)
        encoding_outputs = []

        gammas_encoder, gammas_middle, gammas_decoder = gammas
        betas_encoder, betas_middle, betas_decoder = betas

        for i in range(self.n):
            x = self.encoders[i](x)
            if self.is_encoder_conditioned:
                x = self.film(x, gammas_encoder[i], betas_encoder[i])
            encoding_outputs.append(x)
            x = self.downsamplings[i](x)

        x = self.mid_block(x)
        if self.is_middle_conditioned:
            x = self.film(x, gammas_middle, betas_middle)

        for i in range(self.n):
            x = self.upsamplings[i](x)
            x = torch.cat((x, encoding_outputs[-i - 1]), 1)
            x = self.decoders[i](x)
            if self.is_decoder_conditioned:
                x = self.film(x, gammas_decoder[i], betas_decoder[i])

        return self.last_conv(x)


class DenseCUNet_FiLM_Framework(Dense_CUNet_Framework):

    def __init__(self, n_fft, hop_length, num_frame,
                 spec_type, spec_est_mode,
                 spec2spec,
                 optimizer, lr, auto_lr_schedule,
                 train_loss, val_loss
                 ):

        super(DenseCUNet_FiLM_Framework, self).__init__(
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--control_n_layer', type=int, default=4)
        parser.add_argument('--control_type', type=str, default='dense')
        parser.add_argument('--film_type', type=str, default='complex')

        parser.add_argument('--gamma_activation', type=str, default='identity')
        parser.add_argument('--beta_activation', type=str, default='identity')

        return Dense_CUNet_Framework.add_model_specific_args(parser)
