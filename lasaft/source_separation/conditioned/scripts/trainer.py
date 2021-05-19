import inspect
from warnings import warn

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from pathlib import Path
from pytorch_lightning import Trainer, seed_everything

from lasaft.data.data_provider import DataProvider
from lasaft.source_separation.model_definition import get_class_by_name
from lasaft.utils.functions import mkdir_if_not_exists


def train(param):
    if not isinstance(param, dict):
        args = vars(param)
    else:
        args = param

    framework = get_class_by_name('conditioned_separation', args['model'])
    if args['spec_type'] != 'magnitude':
        args['input_channels'] = 4

    if args['resume_from_checkpoint'] is None:
        if args['seed'] is not None:
            seed_everything(args['seed'])

    model = framework(**args)

    if args['last_activation'] != 'identity' and args['spec_est_mode'] != 'masking':
        warn('Please check if you really want to use a mapping-based spectrogram estimation method '
             'with a final activation function. ')
    ##########################################################

    # -- checkpoint
    ckpt_path = Path(args['ckpt_root_path'])
    mkdir_if_not_exists(ckpt_path)
    ckpt_path = ckpt_path.joinpath(args['model'])
    mkdir_if_not_exists(ckpt_path)
    run_id = args['run_id']
    ckpt_path = ckpt_path.joinpath(run_id)
    mkdir_if_not_exists(ckpt_path)
    save_top_k = args['save_top_k']

    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=save_top_k,
        verbose=False,
        monitor='val_loss',
        save_last=False,
        save_weights_only=args['save_weights_only']
    )
    args['checkpoint_callback'] = checkpoint_callback

    # -- early stop
    patience = args['patience']
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=patience,
        verbose=False
    )
    args['early_stop_callback'] = early_stop_callback

    if args['resume_from_checkpoint'] is not None:
        run_id = run_id + "_resume_" + args['resume_from_checkpoint']
        args['resume_from_checkpoint'] = Path(
            args['ckpt_root_path']).joinpath(
            args['model']).joinpath(
            args['run_id']).joinpath(
            args['resume_from_checkpoint']
        )
        args['resume_from_checkpoint'] = str(args['resume_from_checkpoint'])

    model_name = model.spec2spec.__class__.__name__

    # -- logger setting
    log = args['log']
    if log == 'False':
        args['logger'] = False
    elif log == 'wandb':
        args['logger'] = WandbLogger(project='lasaft_exp', tags=[model_name], offline=False, name=run_id)
        args['logger'].log_hyperparams(model.hparams)
        args['logger'].watch(model, log='all')
    elif log == 'tensorboard':
        raise NotImplementedError
    else:
        args['logger'] = True  # default
        default_save_path = 'etc/lightning_logs'
        mkdir_if_not_exists(default_save_path)

    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = dict((name, args[name]) for name in valid_kwargs if name in args)


    # Trainer
    trainer = Trainer(**trainer_kwargs)
    dataset_args = {'musdb_root': args['musdb_root'],
                    'batch_size': args['batch_size'],
                    'num_workers': args['num_workers'],
                    'pin_memory': args['pin_memory'],
                    'num_frame': args['num_frame'],
                    'hop_length': args['hop_length'],
                    'n_fft': args['n_fft']}

    dp = DataProvider(**dataset_args)
    train_dataset, training_dataloader = dp.get_training_dataset_and_loader()
    valid_dataset, validation_dataloader = dp.get_validation_dataset_and_loader()

    for key in sorted(args.keys()):
        print('{}:{}'.format(key, args[key]))

    if args['auto_lr_find']:
        lr_find = trainer.tuner.lr_find(model,
                                        training_dataloader,
                                        validation_dataloader,
                                        early_stop_threshold=None,
                                        min_lr=1e-5)

        print(f"Found lr: {lr_find.suggestion()}")
        return None

    if args['resume_from_checkpoint'] is not None:
        print('resume from the checkpoint')

    trainer.fit(model, training_dataloader, validation_dataloader)

    return None
