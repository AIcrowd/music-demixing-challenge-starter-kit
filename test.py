#!/usr/bin/env python
# This file is the entrypoint for your submission.
# You can modify this file to include your code or directly call your functions/modules from here.
import shutil
from evaluator.music_demixing import MusicDemixingPredictor


class RandomPredictor(MusicDemixingPredictor):
    """
    PARTICIPANT_TODO:
    You can do any preprocessing required for your codebase here like loading up models into memory, etc.
    """
    def prediction_setup(self):
        # Load your model here.
        # self.separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq')
        pass

    """
    PARTICIPANT_TODO:
    During the evaluation all music files will be provided one by one, along with destination path 
    for saving separated audios.

    NOTE: In case you want to load your model, please do so in `predict_setup` function.
    """
    def prediction(self, mixture_file_path, bass_file_path, drums_file_path, other_file_path, vocals_file_path):
        print("Mixture file is present at following location: %s" % mixture_file_path)

        # Write your prediction code here:
        # [...]
        # estimates = separator(audio)
        # Save the wav files at assigned locations.
        shutil.copyfile(mixture_file_path, bass_file_path)
        shutil.copyfile(mixture_file_path, drums_file_path)
        shutil.copyfile(mixture_file_path, other_file_path)
        shutil.copyfile(mixture_file_path, vocals_file_path)
        print("%s: prediction completed." % mixture_file_path)


if __name__ == "__main__":
    submission = RandomPredictor()
    submission.run()
    print("Successfully generated predictions!")
