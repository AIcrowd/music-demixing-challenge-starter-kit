#!/usr/bin/env python
#
# This file uses openunmix for music demixing.
# It is one of official baseline for Music Demixing challenge.
#
# NOTE: openunmix need checkpoints to be submitted along with your code.
#
# Making submission using openunmix:
# 1. Change the model in `predict.py` to UMXPredictor.
# 2. Run this file locally with `python test_umx.py`.
# 3. Submit your code using git-lfs
#    #> git lfs install
#    #> git lfs track "*.pth"
#    #> git add .gitattributes
#    #> git add models
#
from torch.utils.data import Dataset, DataLoader

from evaluator.music_demixing import MusicDemixingPredictor
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from lasaft.data.musdb_wrapper import SingleTrackSet
from lasaft.pretrained import PreTrainedLaSAFTNet


class LaSAFTPredictor(MusicDemixingPredictor):
    def prediction_setup(self):
        self.model = PreTrainedLaSAFTNet('lasaft_medium_test')
        self.model.eval()
        self.model.freeze()

    def separator(self, audio, rate):
        pass

    def prediction(
            self,
            mixture_file_path,
            bass_file_path,
            drums_file_path,
            other_file_path,
            vocals_file_path,
    ):
        mix, rate = sf.read(mixture_file_path, dtype='float32')
        # mix = np.concatenate((mix, mix), axis=0)
        device = self.model.device
        dataset = SingleTrackSet(mix, self.model.hop_length, self.model.num_frame)

        batch_size = 32
        dataloader = DataLoader(dataset, batch_size=batch_size)

        trim_length = self.model.trim_length
        total_length = mix.shape[0]
        window_length = self.model.hop_length * (self.model.num_frame - 1)
        true_samples = window_length - 2 * trim_length

        results = {0: [], 1: [], 2: [], 3: []}

        with torch.no_grad():
            for mixture, window_ids, offsets in dataloader:
                target_hats = self.model.separate(mixture, offsets)[:, trim_length:-trim_length]  # B, T, 2
                for target_hat, offset in zip(target_hats, offsets):
                    results[offset.item()].append(target_hat)
                # input_conditions.append(input_condition)

        vocals, drums, bass, other = [torch.cat(results[i])[:total_length].cpu().detach().numpy() for i in range(4)]

        target_file_map = {
            "vocals": vocals_file_path,
            "drums": drums_file_path,
            "bass": bass_file_path,
            "other": other_file_path,
        }

        for target, target_name in zip([vocals, drums, bass, other], ['vocals', 'drums', 'bass', 'other']):
            sf.write(target_file_map[target_name], mix, samplerate=44100)


if __name__ == "__main__":
    submission = LaSAFTPredictor()
    submission.run()
    print("Successfully generated predictions!")
