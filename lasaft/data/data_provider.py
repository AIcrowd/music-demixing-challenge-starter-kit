from argparse import ArgumentParser
from torch.utils.data import DataLoader
from lasaft.data.musdb_wrapper import MusdbTrainSet, MusdbValidSetWithGT, MusdbTestSetWithGT


class DataProvider(object):

    @staticmethod
    def add_data_provider_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--musdb_root', type=str, default='etc/musdb18_samples_wav/')

        return parser

    def __init__(self, musdb_root, batch_size, num_workers, pin_memory, n_fft, hop_length, num_frame):
        self.musdb_root = musdb_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_frame = num_frame
        self.hop_length = hop_length
        self.n_fft = n_fft

    def get_training_dataset_and_loader(self):
        training_set = MusdbTrainSet(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)

        loader = DataLoader(training_set, shuffle=True, batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)

        return training_set, loader

    def get_validation_dataset_and_loader(self):
        validation_set = MusdbValidSetWithGT(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)

        loader = DataLoader(validation_set, shuffle=False, batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)

        return validation_set, loader

    def get_test_dataset_and_loader(self):
        test_set = MusdbTestSetWithGT(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)

        loader = DataLoader(test_set, shuffle=False, batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)

        return test_set, loader
