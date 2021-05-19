from argparse import ArgumentParser

import torch

from lasaft.source_separation.conditioned.cunet.dcun_base import Dense_CUNet, Dense_CUNet_Framework
from lasaft.source_separation.sub_modules.control_models import pocm_control_model, dense_control_block
from lasaft.utils.PoCM_utils import Pocm_Matmul, Pocm_naive


class DenseCUNet_GPoCM(Dense_CUNet):

    def __init__(self,
                 n_fft,
                 input_channels, internal_channels,
                 n_blocks, n_internal_layers,
                 mk_block_f, mk_ds_f, mk_us_f,
                 first_conv_activation, last_activation,
                 t_down_layers, f_down_layers,
                 # Conditional Mechanism #
                 control_vector_type, control_input_dim, embedding_dim, condition_to,
                 control_type, control_n_layer, pocm_type, pocm_norm
                 ):

        super(DenseCUNet_GPoCM, self).__init__(
            n_fft,
            input_channels, internal_channels,
            n_blocks, n_internal_layers,
            mk_block_f, mk_ds_f, mk_us_f,
            first_conv_activation, last_activation,
            t_down_layers, f_down_layers,
            # Conditional Mechanism #
            control_vector_type, control_input_dim, embedding_dim, condition_to
        )

        # select PoCM implementation:
        # both yield the same outputs, but 'matmul' is faster with gpus since it does not use loops.
        assert pocm_type in ['naive', 'matmul']
        self.pocm = Pocm_naive if pocm_type == 'naive' else Pocm_Matmul

        # Select normalization methods for PoCM
        assert pocm_norm in [None, 'batch_norm']

        # Make condition generator
        if control_type == "dense":
            self.condition_generator = pocm_control_model(
                dense_control_block(embedding_dim, control_n_layer),
                n_blocks, internal_channels,
                pocm_to=condition_to,
                pocm_norm=pocm_norm
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
                g = self.pocm(x, gammas_encoder[i], betas_encoder[i]).sigmoid()
                x = g * x
            encoding_outputs.append(x)
            x = self.downsamplings[i](x)

        x = self.mid_block(x)
        if self.is_middle_conditioned:
            g = self.pocm(x, gammas_middle, betas_middle).sigmoid()
            x = g * x

        for i in range(self.n):
            x = self.upsamplings[i](x)
            x = torch.cat((x, encoding_outputs[-i - 1]), 1)
            x = self.decoders[i](x)
            if self.is_decoder_conditioned:
                g = self.pocm(x, gammas_decoder[i], betas_decoder[i]).sigmoid()
                x = g * x
        return self.last_conv(x)


class DenseCUNet_GPoCM_Framework(Dense_CUNet_Framework):

    def __init__(self, n_fft, hop_length, num_frame,
                 spec_type, spec_est_mode,
                 spec2spec,
                 optimizer, lr, auto_lr_schedule,
                 train_loss, val_loss
                 ):

        super(DenseCUNet_GPoCM_Framework, self).__init__(
            n_fft, hop_length, num_frame,
            spec_type, spec_est_mode,
            spec2spec,
            optimizer, lr, auto_lr_schedule,
            train_loss, val_loss
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--control_n_layer', type=int, default=4)
        parser.add_argument('--control_type', type=str, default='dense')

        parser.add_argument('--pocm_type', type=str, default='matmul')
        parser.add_argument('--pocm_norm', type=str, default='batch_norm')

        return Dense_CUNet_Framework.add_model_specific_args(parser)
