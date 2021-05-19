import os
import requests

from lasaft.source_separation.conditioned.cunet.models.dcun_tfc_gpocm_lasaft import DCUN_TFC_GPoCM_LaSAFT_Framework
from lasaft.source_separation.conditioned.cunet.models.dcun_tfc_gpocm_lightsaft import \
    DCUN_TFC_GPoCM_LightSAFT_Framework


def __define_large_params__(is_light):
    args = {}

    # FFT params
    args['n_fft'] = 4096
    args['hop_length'] = 1024
    args['num_frame'] = 128

    # SVS Framework
    args['spec_type'] = 'complex'
    args['spec_est_mode'] = 'mapping'

    # Other Hyper-params
    args['optimizer'] = 'adam'
    args['lr'] = 0.0001
    args['train_loss'] = 'spec_mse'
    args['val_loss'] = 'raw_l1'

    # DenseNet Hyper-params
    args['n_blocks'] = 9
    args['input_channels'] = 4
    args['internal_channels'] = 24
    args['first_conv_activation'] = 'relu'
    args['last_activation'] = 'identity'
    args['t_down_layers'] = None
    args['f_down_layers'] = None

    # TFC_TDF Block's Hyper-params
    args['n_internal_layers'] = 5
    args['kernel_size_t'] = 3
    args['kernel_size_f'] = 3
    args['tfc_tdf_activation'] = 'relu'
    args['bn_factor'] = 16
    args['min_bn_units'] = 16
    args['tfc_tdf_bias'] = False
    args['num_tdfs'] = 6
    args['dk'] = 32

    args['control_vector_type'] = 'embedding'
    args['control_input_dim'] = 4
    args['embedding_dim'] = 64
    args['condition_to'] = 'decoder'

    args['control_n_layer'] = 4
    args['control_type'] = 'dense'
    args['pocm_type'] = 'matmul'
    args['pocm_norm'] = 'batch_norm'

    args['auto_lr_schedule'] = True
    return DCUN_TFC_GPoCM_LightSAFT_Framework(**args) if is_light else DCUN_TFC_GPoCM_LaSAFT_Framework(**args)


def __define_small_params__(is_light=False):
    args = {}

    # FFT params
    args['n_fft'] = 2048
    args['hop_length'] = 1024
    args['num_frame'] = 256

    # SVS Framework
    args['spec_type'] = 'complex'
    args['spec_est_mode'] = 'mapping'

    # Other Hyper-params
    args['optimizer'] = 'adam'
    args['lr'] = 0.0005
    args['train_loss'] = 'spec_mse'
    args['val_loss'] = 'raw_l1'

    # DenseNet Hyper-params
    args['n_blocks'] = 9
    args['input_channels'] = 4
    args['internal_channels'] = 16
    args['first_conv_activation'] = 'relu'
    args['last_activation'] = 'identity'
    args['t_down_layers'] = None
    args['f_down_layers'] = None

    # TFC_TDF Block's Hyper-params
    args['n_internal_layers'] = 5
    args['kernel_size_t'] = 3
    args['kernel_size_f'] = 3
    args['tfc_tdf_activation'] = 'relu'
    args['bn_factor'] = 16
    args['min_bn_units'] = 16
    args['tfc_tdf_bias'] = False
    args['num_tdfs'] = 16
    args['dk'] = 64

    args['control_vector_type'] = 'embedding'
    args['control_input_dim'] = 4
    args['embedding_dim'] = 64
    args['condition_to'] = 'decoder'

    args['control_n_layer'] = 4
    args['control_type'] = 'dense'
    args['pocm_type'] = 'matmul'
    args['pocm_norm'] = 'batch_norm'

    args['auto_lr_schedule'] = False
    return DCUN_TFC_GPoCM_LightSAFT_Framework(**args) if is_light else DCUN_TFC_GPoCM_LaSAFT_Framework(**args)


def PreTrainedLaSAFTNet(model_name='lasaft_large_2020'):
    assert model_name in ['lasaft_large_2020', 'lasaft_large_2021', 'lasaft_medium_test']
    ckpt = model_name + '.ckpt'

    if 'test' in model_name:
        pass
    elif not os.path.exists(ckpt):
        print('no cached checkpoint found.\nautomatic download!')
        url = 'http://intelligence.korea.ac.kr/assets/' + model_name + '.ckpt'
        response = requests.get(url, stream=True)
        print('successfully downloaded the pretrained model.')

    if 'medium_' in model_name:
        return __define_small_params__()
        pass
    elif 'large' in model_name:
        model = __define_large_params__()
        return model.load_from_checkpoint(ckpt)
    else:
        raise ModuleNotFoundError


def PreTrainedLightSAFTNet(model_name):
    assert model_name in ['lightsaft_small_2020']
    ckpt = model_name + '.ckpt'

    if 'test' in model_name:
        pass
    elif not os.path.exists(ckpt):
        print('no cached checkpoint found.\nautomatic download!')
        url = 'http://intelligence.korea.ac.kr/assets/' + model_name + '.ckpt'
        response = requests.get(url, stream=True)
        with open(model_name + '.ckpt', 'wb') as f:
            f.write(response.content)

        print('successfully downloaded the pretrained model.')

    if 'small' in model_name:
        model = __define_small_params__(is_light=True)
        return model.load_from_checkpoint(ckpt)
    elif 'large' in model_name:
        model = __define_large_params__()
        return model.load_from_checkpoint(ckpt)
    else:
        raise ModuleNotFoundError
