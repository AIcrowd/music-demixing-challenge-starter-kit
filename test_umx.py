#!/usr/bin/env python
# This file is the entrypoint for your submission.
# You can modify this file to include your code or directly call your functions/modules from here.
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
