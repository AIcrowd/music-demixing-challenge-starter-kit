#!/usr/bin/env python
#######################
# openunmix need checkpoints to be submitted along with your code.
# to do so, run the test_umx.py locally, followed by copying ~/.cache/torch/hub to repository as .cache folder
#######################

from evaluator.music_demixing import MusicDemixingPredictor
import torch
import torchaudio
from openunmix import data, predict

class UMXPredictor(MusicDemixingPredictor):
    def prediction_setup(self):
        # Load your model here.
        self.separator = torch.hub.load("sigsep/open-unmix-pytorch", "umxhq")

    def prediction(
        self,
        music_name,
        mixture_file_path,
        bass_file_path,
        drums_file_path,
        other_file_path,
        vocals_file_path,
    ):

        audio, rate = data.load_audio(mixture_file_path)
        estimates = predict.separate(
            audio=audio, rate=rate, separator=self.separator
        )

        target_file_map = {
            "vocals": vocals_file_path,
            "drums": drums_file_path,
            "bass": bass_file_path,
            "other": other_file_path,
        }
        for target, path in target_file_map.items():
            torchaudio.save(
                path,
                torch.squeeze(estimates[target]),
                sample_rate=self.separator.sample_rate,
            )
        print("%s: prediction completed." % music_name)


if __name__ == "__main__":
    submission = UMXPredictor()
    submission.run()
    print("Successfully generated predictions!")
