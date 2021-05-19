import inspect
from warnings import warn

from pytorch_lightning.loggers import WandbLogger

from lasaft.data.data_provider import DataProvider
from lasaft.source_separation.model_definition import get_class_by_name
from lasaft.utils.functions import mkdir_if_not_exists
from pathlib import Path
from pytorch_lightning import Trainer


def eval(param):

    if not isinstance(param, dict):
        args = vars(param)
    else:
        args = param

    for key in args.keys():
        if args[key] == 'None':
            args[key] = None

    if args['gpu_index'] is not None:
        args['gpus'] = str(args['gpu_index'])

    # MODEL
    ##########################################################
    # # # get framework
    framework = get_class_by_name('conditioned_separation', args['model'])
    if args['spec_type'] != 'magnitude':
        args['input_channels'] = 4
    # # # Model instantiation
    from copy import deepcopy as c
    model_args = c(args)
    model = framework(**model_args)
    ##########################################################

    # Trainer Definition

    # -- checkpoint
    ckpt_path = Path(args['ckpt_root_path']).joinpath(args['model']).joinpath(args['run_id'])
    ckpt_path = '{}/{}'.format(str(ckpt_path), args['epoch'])

    # -- logger setting
    log = args['log']
    if log == 'False':
        args['logger'] = False
        args['checkpoint_callback'] = False
        args['early_stop_callback'] = False
    elif log == 'wandb':
        args['logger'] = WandbLogger(project='lasaft_exp', tags=args['model'], offline=False,
                                     name=args['run_id'] + '_eval_' + args['epoch'].replace('=','_'))
        args['logger'].log_hyperparams(model.hparams)
        args['logger'].watch(model, log='all')
    elif log == 'tensorboard':
        raise NotImplementedError
    else:
        args['logger'] = True  # default
        default_save_path = 'etc/lightning_logs'
        mkdir_if_not_exists(default_save_path)

    # Trainer
    if isinstance(args['gpus'], int):
        if args['gpus'] > 1:
            warn('# gpu and num_workers should be 1, Not implemented: museval for distributed parallel')
            args['gpus'] = 1
            args['distributed_backend'] = None

    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = dict((name, args[name]) for name in valid_kwargs if name in args)

    # DATASET
    ##########################################################
    dataset_args = {'musdb_root': args['musdb_root'],
                    'batch_size': args['batch_size'],
                    'num_workers': args['num_workers'],
                    'pin_memory': args['pin_memory'],
                    'num_frame': args['num_frame'],
                    'hop_length': args['hop_length'],
                    'n_fft': args['n_fft']}
    dp = DataProvider(**dataset_args)
    ##########################################################

    trainer_kwargs['precision'] = 32
    trainer = Trainer(**trainer_kwargs)
    _, test_data_loader = dp.get_test_dataset_and_loader()
    model = model.load_from_checkpoint(ckpt_path)

    trainer.test(model, test_data_loader)

    return None