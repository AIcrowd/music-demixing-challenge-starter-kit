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
# 3. Copy the local torch hub cache to this repository:
#    #> mkdir -p .cache/torch/hub
#    #> cp -r ~/.cache/torch/hub .cache/torch/
# 4. Submit your code using git-lfs
#    #> git lfs install
#    #> git lfs track "*.pth"
#    #> git add .gitattributes
#    #> git add .cache
#

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
        mixture_file_path,
        bass_file_path,
        drums_file_path,
        other_file_path,
        vocals_file_path,
    ):

        audio, rate = data.load_audio(mixture_file_path)
        # mixture rate is 44100 Hz
        # umx .separate includes resampling to model samplerate
        # here, nothing is done as model samplerate == 44100
        estimates = predict.separate(audio=audio, rate=rate, separator=self.separator)

        target_file_map = {
            "vocals": vocals_file_path,
            "drums": drums_file_path,
            "bass": bass_file_path,
            "other": other_file_path,
        }
        for target, path in target_file_map.items():
            if rate != self.separator.sample_rate:
                # in case the estimate sample rate is different
                # to mixture (44100) samplerate we need to resample
                print("resample to mixture sample rate")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=self.separator.sample_rate,
                    new_freq=rate,
                    resampling_method="sinc_interpolation",
                )
                target_estimate = torch.squeeze(resampler(estimates[target]))
            else:
                target_estimate = torch.squeeze(estimates[target])
            torchaudio.save(
                path,
                target_estimate,
                sample_rate=rate,
            )
        print("%s: prediction completed." % mixture_file_path)


if __name__ == "__main__":
    submission = UMXPredictor()
    submission.run()
    print("Successfully generated predictions!")
