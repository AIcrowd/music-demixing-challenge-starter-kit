![Music Demixing Banner](https://images.aicrowd.com/raw_images/challenges/social_media_image_file/777/8be36d177c2b161d7944.jpg)

# [Music Demixing Challenge ](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021)- Starter Kit 
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the Music Demixing Challenge **Submission template and Starter kit**! Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!

> **NOTE:** 
If you are resource-constrained or would not like to setup everything in your system, you can make your submission from inside Google Colab too. [**Check out the beat version of the Notebook.**](https://colab.research.google.com/drive/14FpktUXysnjIL165hU3rTUKPHo4-YRPh?usp=sharing)



# Table of Contents

1. [Competition Procedure](#competition-procedure)
2. [How to access and use dataset](#how-to-access-and-use-dataset)
3. [How to start participating](#how-to-start-participating)
4. [How do I specify my software runtime / dependencies?](#how-do-i-specify-my-software-runtime-dependencies-)
5. [What should my code structure be like ?](#what-should-my-code-structure-be-like-)
6. [How to make submission](#how-to-make-submission)
7. [Other concepts](#other-concepts)
8. [Important links](#-important-links)


<p style="text-align:center"><img style="text-align:center" src="https://images.aicrowd.com/uploads/ckeditor/pictures/401/content_image.png"></p>


#  Competition Procedure

The Music Demixing (MDX) Challenge is an opportunity for researchers and machine learning enthusiasts to test their skills by creating a system able to perform audio source separation.

In this challenge, you will train your models locally and then upload them to AIcrowd (via git) to be evaluated. 

**The following is a high level description of how this process works**

![](https://i.imgur.com/xzQkwKV.jpg)

1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021).
2. **Clone** this repo and start developing your solution.
3. **Train** your models for audio seperation and write prediction code in `test.py`.
4. [**Submit**](#how-to-submit-a-model) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-submit-a-model). The automated evaluation setup will evaluate the submissions against the test dataset to compute and report the metrics on the leaderboard of the competition.

# How to access and use the dataset

You are allowed to train your system either exclusively on the training part of
[MUSDB18-HQ dataset](https://zenodo.org/record/3338373) or you can use your choice of data.
According to the dataset used, you will be eligible for different leaderboards.

👉 [Download MUSDB18-HQ dataset](https://zenodo.org/record/3338373)

In case you are using external dataset, please mention it in your `aicrowd.json`.
```bash
{
  [...],
  "external_dataset_used": true
}
```

The MUSDB18 dataset contains `150` songs (`100` songs in `train` and `50` songs in `test`) together with their seperations
in the following manner:

```bash
|
├── train
│   ├── A Classic Education - NightOwl
│   │   ├── bass.wav
│   │   ├── drums.wav
│   │   ├── mixture.wav
│   │   ├── other.wav
│   │   └── vocals.wav
│   └── ANiMAL - Clinic A
│       ├── bass.wav
│       ├── drums.wav
│       ├── mixture.wav
│       ├── other.wav
│       └── vocals.wav
[...]
```

Here the `mixture.wav` file is the original music on which you need to do audio source seperation.<br>
While `bass.wav`, `drums.wav`, `other.wav` and `vocals.wav` contain files for your training purposes.<br>
<b>Please note again:</b> To be eligible for Leaderboard A, you are only allowed to train on the songs in `train`.


# How to start participating

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2.  **Clone the repository**

    ```
    git clone git@github.com:AIcrowd/music-demixing-challenge-starter-kit.git
    ```

3. **Install** competition specific dependencies!
    ```
    cd music-demixing-challenge-starter-kit
    pip3 install -r requirements.txt
    ```

4. Try out random prediction codebase present in `test.py`.


## How do I specify my software runtime / dependencies ?

We accept submissions with custom runtime, so you don't need to worry about which libraries or framework to pick from.

The configuration files typically include `requirements.txt` (pypi packages), `environment.yml` (conda environment), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the 👉 [RUNTIME.md](/docs/RUNTIME.md) file.

## What should my code structure be like ?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:

```
.
├── aicrowd.json           # Submission meta information - like your username
├── apt.txt                # Packages to be installed inside docker image
├── data                   # Your local dataset copy - you don't need to upload it (read DATASET.md)
├── requirements.txt       # Python packages to be installed
├── test.py                # IMPORTANT: Your testing/prediction code, must be derived from MusicDemixingPredictor (example in test.py)
└── utility                # The utility scripts to provide smoother experience to you.
    ├── docker_build.sh
    ├── docker_run.sh
    ├── environ.sh
    └── verify_or_download_data.sh
```

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!** 

The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "evaluations-api-music-demixing",
  "authors": ["your-aicrowd-username"],
  "description": "(optional) description about your awesome agent",
  "external_dataset_used": false
}
```

This JSON is used to map your submission to the challenge - so please remember to use the correct `challenge_id` as specified above.

## How to make submission

👉 [SUBMISSION.md](/docs/SUBMISSION.md)

**Best of Luck** :tada: :tada:

# Other Concepts

## Time constraints

You need to make sure that your model can do audio seperation for each song within 4 minutes, otherwise the submission will be marked as failed.

## Local Run

👉 [LOCAL_RUN.md](/docs/LOCAL_RUN.md)

## Contributing

🙏 You can share your solutions or any other baselines by contributing directly to this repository by opening merge request.

- Add your implemntation as `test_<approach-name>.py`
- Test it out using `python test_<approach-name>.py`
- Add any documentation for your approach at top of your file.
- Import it in `predict.py`
- Create merge request! 🎉🎉🎉 

## Contributors

- [Stefan Uhlich](https://www.aicrowd.com/participants/StefanUhlich)
- [Fabian-Robert Stöter](https://www.aicrowd.com/participants/faroit)
- [Shivam Khandelwal](https://www.aicrowd.com/participants/shivam)

# 📎 Important links


💪 &nbsp;Challenge Page: https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021

🗣️ &nbsp;Discussion Forum: https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021/discussion

🏆 &nbsp;Leaderboard: https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021/leaderboards
