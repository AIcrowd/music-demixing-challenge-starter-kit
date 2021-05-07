#!/usr/bin/env python
#
# This file uses CrossNet-UMX (X-UMX) for music demixing.
# It is one of the official baselines for the Music Demixing challenge.
#
# NOTE: X-UMX needs the model to be submitted along with your code.
#
# Making submission using X-UMX:
# 1. Change the model in `predict.py` to XUMXPredictor.
# 2. Download the pre-trained model from Zenodo into the folder `./models`
#    #> mkdir models
#    #> cd models
#    #> wget https://zenodo.org/record/4740378/files/pretrained_xumx_musdb18HQ.pth
# 3. Submit your code using git-lfs
#    #> git lfs install
#    #> git lfs track "*.pth"
#    #> git add .gitattributes
#    #> git add models
#

from asteroid.models import XUMX
from asteroid.complex_nn import torch_complex_from_magphase
import norbert
import numpy as np
import scipy
import soundfile as sf
import torch

from evaluator.music_demixing import MusicDemixingPredictor


# Inverse STFT - taken from
#    https://github.com/asteroid-team/asteroid/blob/master/egs/musdb18/X-UMX/eval.py
def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2), rate, nperseg=n_fft, noverlap=n_fft - n_hopsize, boundary=True
    )
    return audio

# Separation function - taken from
#    https://github.com/asteroid-team/asteroid/blob/master/egs/musdb18/X-UMX/eval.py
def separate(
    audio,
    x_umx_target,
    instruments,
    niter=1,
    softmask=False,
    alpha=1.0,
    residual_model=False,
    device="cpu",
):
    """
    Performing the separation on audio input
    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio
    x_umx_target: asteroid.models
        X-UMX model used for separating
    instruments: list
        The list of instruments, e.g., ["bass", "drums", "vocals"]
    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.
    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False
    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0
    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False
    device: str
        set torch device. Defaults to `cpu`.
    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary with all estimates obtained by the separation model.
    """

    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)

    source_names = []
    V = []

    masked_tf_rep, _ = x_umx_target(audio_torch)
    # shape: (Sources, frames, batch, channels, fbin)

    for j, target in enumerate(instruments):
        Vj = masked_tf_rep[j, Ellipsis].cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj ** alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, Ellipsis])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    # convert to complex numpy type
    tmp = x_umx_target.encoder(audio_torch)
    X = torch_complex_from_magphase(tmp[0].permute(1, 2, 3, 0), tmp[1])
    X = X.detach().cpu().numpy()
    X = X[0].transpose(2, 1, 0)

    if residual_model or len(instruments) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += ["residual"] if len(instruments) > 1 else ["accompaniment"]

    Y = norbert.wiener(V, X.astype(np.complex128), niter, use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            rate=x_umx_target.sample_rate,
            n_fft=x_umx_target.in_chan,
            n_hopsize=x_umx_target.n_hop,
        )
        estimates[name] = audio_hat.T

    return estimates


class XUMXPredictor(MusicDemixingPredictor):
    def prediction_setup(self):
        # Load your model here and put it into `evaluation` mode
        self.separator = XUMX.from_pretrained("./models/pretrained_xumx_musdb18HQ.pth")
        self.separator.eval()

    def prediction(
        self,
        mixture_file_path,
        bass_file_path,
        drums_file_path,
        other_file_path,
        vocals_file_path,
    ):

        # Step 1: Load mixture
        x, rate = sf.read(mixture_file_path)  # mixture is stereo with sample rate of 44.1kHz

        # Step 2: Pad mixture to compensate STFT truncation
        x_padded = np.pad(x, ((0, 1024), (0, 0)))

        # Step 3: Perform separation
        estimates = separate(
            x_padded,
            self.separator,
            self.separator.sources
        )

        # Step 4: Truncate to orignal length
        for target in estimates:
            estimates[target] = estimates[target][:x.shape[0], :]

        # Step 5: Store results
        target_file_map = {
            "vocals": vocals_file_path,
            "drums": drums_file_path,
            "bass": bass_file_path,
            "other": other_file_path,
        }
        for target, path in target_file_map.items():
            sf.write(
                path,
                estimates[target],
                rate
            )


if __name__ == "__main__":
    submission = XUMXPredictor()
    submission.run()
    print("Successfully generated predictions!")
