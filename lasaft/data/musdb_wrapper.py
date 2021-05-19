import random
from abc import ABCMeta
from pathlib import Path
from torch.utils.data import Dataset
from lasaft.utils.fourier import get_trim_length

import musdb
import numpy as np
import soundfile
import torch


def check_musdb_valid(musdb_train):
    if len(musdb_train) > 0:
        pass
    else:
        print('It seems like you used wrong path for musdb18 dataset')
        raise NotImplemented  # TODO: Exception handling


class MusdbWrapperDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, musdb_root, subset, n_fft, hop_length, num_frame):

        musdb_root = Path(musdb_root)
        self.root = musdb_root.joinpath(subset)

        if subset == 'test':
            self.musdb_reference = musdb.DB(root=musdb_root, subsets='test', is_wav=True)
        elif subset == 'train':
            self.musdb_reference = musdb.DB(root=musdb_root, subsets='train', split='train', is_wav=True)
        elif subset == 'valid':
            self.musdb_reference = musdb.DB(root=musdb_root, subsets='train', split='valid', is_wav=True)
        else:
            raise ModuleNotFoundError

        check_musdb_valid(self.musdb_reference)

        self.target_names = self.source_names = ['vocals', 'drums', 'bass', 'other']

        self.num_tracks = len(self.musdb_reference)
        self.num_targets = len(self.target_names)
        # cache wav file dictionary
        self.wav_dict = {i: {s: self.musdb_reference.tracks[i].sources[s].path for s in self.source_names}
                         for i in range(self.num_tracks)}

        for i in range(self.num_tracks):
            self.wav_dict[i]['mixture'] = self.musdb_reference[i].path

        self.lengths = [self.get_audio(i, 'vocals').shape[0] for i in range(self.num_tracks)]
        self.window_length = hop_length * (num_frame - 1)

    def __len__(self) -> int:
        return self.num_tracks * self.num_targets

    def get_audio(self, idx, target_name, pos=0, length=None):
        arg_dicts = {
            'file': self.wav_dict[idx][target_name],
            'start': pos,
            'dtype': 'float32'
        }

        if length is not None:
            arg_dicts['stop'] = pos + length

        return soundfile.read(**arg_dicts)[0]


class MusdbTrainSet(MusdbWrapperDataset):

    def __init__(self,
                 musdb_root='etc/musdb18_samples_wav/',
                 n_fft=2048,
                 hop_length=1024,
                 num_frame=64):
        super(MusdbTrainSet, self).__init__(musdb_root, 'train', n_fft, hop_length, num_frame)

    def __len__(self):
        return sum([length // self.window_length for length in self.lengths]) * len(self.target_names)

    def __getitem__(self, whatever):
        source_sample = {target: self.get_random_audio_sample(target) for target in self.source_names}
        rand_target = np.random.choice(self.target_names)

        mixture = sum(source_sample.values())
        target = source_sample[rand_target]

        input_condition = np.array(self.source_names.index(rand_target), dtype=np.long)

        return torch.from_numpy(mixture), torch.from_numpy(target), torch.tensor(input_condition, dtype=torch.long)

    def get_random_audio_sample(self, target_name):
        return self.get_audio_sample(random.randint(0, self.num_tracks - 1), target_name)

    def get_audio_sample(self, idx, target_name):
        length = self.lengths[idx] - self.window_length
        start_position = random.randint(0, length - 1)
        return self.get_audio(idx, target_name, start_position, self.window_length)


class MusdbEvalSet(MusdbWrapperDataset):

    def __init__(self, musdb_root, eval_type, n_fft, hop_length, num_frame):

        super(MusdbEvalSet, self).__init__(musdb_root, eval_type, n_fft, hop_length, num_frame)

        self.hop_length = hop_length
        self.trim_length = get_trim_length(self.hop_length)
        self.true_samples = self.window_length - 2 * self.trim_length

        import math
        num_chunks = [math.ceil(length / self.true_samples) for length in self.lengths]
        self.acc_chunk_final_ids = [sum(num_chunks[:i + 1]) for i in range(self.num_tracks)]

        file_name = 'linear_mixture.wav'
        for i in range(self.num_tracks):
            self.wav_dict[i]['linear_mixture'] = self.wav_dict[i]['vocals'][:-10] + file_name

    def __len__(self):
        return self.acc_chunk_final_ids[-1] * len(self.target_names)

    def __getitem__(self, idx):

        target_offset = idx % len(self.target_names)
        idx = idx // len(self.target_names)

        target_name = self.target_names[target_offset]

        mixture, mixture_idx, offset = self.get_mixture_sample(idx)

        input_condition = np.array(target_offset, dtype=np.long)

        mixture = torch.from_numpy(mixture)
        input_condition = torch.tensor(input_condition, dtype=torch.long)
        window_offset = offset // self.true_samples

        return mixture, mixture_idx, window_offset, input_condition, target_name

    def get_mixture_sample(self, idx):
        mixture_idx, start_pos = self.idx_to_track_offset(idx)
        length = self.true_samples
        left_padding_num = right_padding_num = self.trim_length
        mixture_length = self.lengths[mixture_idx]
        if start_pos + length > mixture_length:  # last
            right_padding_num += self.true_samples - (mixture_length - start_pos)
            length = None

        mixture = self.get_audio(mixture_idx, 'linear_mixture', start_pos, length)

        mixture = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), mixture,
                                  np.zeros((right_padding_num, 2), dtype=np.float32)), 0)
        return mixture, mixture_idx, start_pos

    def idx_to_track_offset(self, idx):
        for i, last_chunk in enumerate(self.acc_chunk_final_ids):
            if idx < last_chunk:
                if i != 0:
                    offset = (idx - self.acc_chunk_final_ids[i - 1]) * self.true_samples
                else:
                    offset = idx * self.true_samples
                return i, offset

        return None, None


def MusdbValidSet(musdb_root='etc/musdb18_samples_wav/',
                  n_fft=2048,
                  hop_length=1024,
                  num_frame=64):
    return MusdbEvalSet(musdb_root, 'valid', n_fft, hop_length, num_frame)


def MusdbTestSet(musdb_root='etc/musdb18_samples_wav/',
                 n_fft=2048,
                 hop_length=1024,
                 num_frame=64):
    return MusdbEvalSet(musdb_root, 'test', n_fft, hop_length, num_frame)


class MusdbEvalSetWithGT(MusdbEvalSet):

    def __init__(self, musdb_root, eval_type, n_fft, hop_length, num_frame):
        super(MusdbEvalSetWithGT, self).__init__(musdb_root, eval_type, n_fft, hop_length, num_frame)

    def __getitem__(self, idx):
        target_offset = idx % len(self.target_names)
        idx = idx // len(self.target_names)

        target_name = self.target_names[target_offset]

        mixture, target, mixture_idx, offset = self.get_mixture_sample_with_GT(idx, target_name)

        input_condition = np.array(target_offset, dtype=np.long)

        mixture = torch.from_numpy(mixture)
        input_condition = torch.tensor(input_condition, dtype=torch.long)
        window_offset = offset // self.true_samples

        return mixture, target, mixture_idx, window_offset, input_condition, target_name

    def get_mixture_sample_with_GT(self, idx, target_name):
        mixture_idx, start_pos = self.idx_to_track_offset(idx)
        length = self.true_samples
        left_padding_num = right_padding_num = self.trim_length
        mixture_length = self.lengths[mixture_idx]
        if start_pos + length > mixture_length:  # last
            right_padding_num += self.true_samples - (mixture_length - start_pos)
            length = None

        mixture = self.get_audio(mixture_idx, 'linear_mixture', start_pos, length)
        target = self.get_audio(mixture_idx, target_name, start_pos, length)

        mixture = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), mixture,
                                  np.zeros((right_padding_num, 2), dtype=np.float32)), 0)
        target = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), target,
                                 np.zeros((right_padding_num, 2), dtype=np.float32)), 0)

        return mixture, target, mixture_idx, start_pos


def MusdbValidSetWithGT(musdb_root='etc/musdb18_samples_wav/',
                        n_fft=2048,
                        hop_length=1024,
                        num_frame=64):
    return MusdbEvalSetWithGT(musdb_root, 'valid', n_fft, hop_length, num_frame)


def MusdbTestSetWithGT(musdb_root='etc/musdb18_samples_wav/',
                       n_fft=2048,
                       hop_length=1024,
                       num_frame=64):
    return MusdbEvalSetWithGT(musdb_root, 'test', n_fft, hop_length, num_frame)


class SingleTrackSet(Dataset):

    def __init__(self, track, hop_length, num_frame):

        assert len(track.shape) == 2
        assert track.shape[1] == 2  # check stereo audio

        self.hop_length = hop_length
        self.window_length = hop_length * (num_frame - 1)
        self.trim_length = get_trim_length(self.hop_length, 8192)

        self.true_samples = self.window_length - 2 * self.trim_length
        self.lengths = [track.shape[0]]
        self.source_names = ['vocals', 'drums', 'bass', 'other']

        pad = self.true_samples - track.shape[0] % self.true_samples
        self.mix_p = np.concatenate((np.zeros((self.trim_length, 2)),
                                     track,
                                     np.zeros((pad, 2)),
                                     np.zeros((self.trim_length, 2))), 0)
        self.mix_p = self.mix_p.astype('float32')

        import math
        self.num_chunks = math.ceil(track.shape[0] / self.true_samples)

    def __len__(self):
        return self.num_chunks * 4

    def __getitem__(self, idx):

        target_offset = idx % 4
        idx = idx // 4

        start_pos = idx * self.true_samples

        mixture = self.mix_p[start_pos:start_pos+self.window_length]

        return torch.from_numpy(mixture), idx, target_offset

    def get_audio(self, pos=0, length=None):

        track = self.cached

        return track[pos:pos + length] if length is not None else track[pos:]
