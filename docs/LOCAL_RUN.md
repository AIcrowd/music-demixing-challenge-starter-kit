
# Local Run

You can simply run `python test.py` for running your submissions locally.
Few default environment variables:
- `TEST_DATASET_PATH` (default: `data/test/`): path to the test dataset folder.
- `RESULTS_DATASET_PATH` (default: `data/results/`): path to the results dataset folder.
- `INFERENCE_SETUP_TIMEOUT_SECONDS` (default: `900` seconds): timeout for your `predict_setup` function.
- `INFERENCE_PER_MUSIC_TIMEOUT_SECONDS` (default: `240` seconds): timeout for your `predict` function.

```bash
python test.py
```

Directory structure after running will look something like:

```
.
├── test
│   ├── SS_008
│   │   └── mixture.wav
│   └── SS_018
│       └── mixture.wav
└── results
    ├── SS_008
    │   ├── bass.wav
    │   ├── drums.wav
    │   ├── mixture.wav
    │   ├── other.wav
    │   └── vocals.wav
    └── SS_018
        ├── bass.wav
        ├── drums.wav
        ├── mixture.wav
        ├── other.wav
        └── vocals.wav
```

## Scoring (local)

You can also calculate scores for your local evaluation by running:

```bash
python score.py
```

This will compare files present in `test/` folder with `results/` folder for SDR calculation.

```bash
def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num  / den)
```
