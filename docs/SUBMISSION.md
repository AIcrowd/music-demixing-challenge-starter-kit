# Making submission

This file will help you in making your first submission.


## Submission Entrypoint (where you write your code!)

The evaluator will execute `run.sh` for generating predictions, so please remember to include it in your submission!

The inline documentation of `test.py` will guide you with interfacing with the codebase properly. You can check TODOs inside it to learn about the functions you need to implement.

You can modify the existing `test.py` OR copy it (to say `your_code.py`) and change it.

The file should adhere to the following constraints:
1. Derived from `MusicDemixingPredictor` class
2. `inference` function needs to be implemented

Once done, you can specify the file you want to run in `run.sh` (by default, it is `test.py` i.e. Random Predictions).

## IMPORTANT: Saving Models before submission!

Before you submit make sure that you have saved your models, which are needed by your inference code.
In case your files are larger in size you can use `git-lfs` to upload them. More details [here](https://discourse.aicrowd.com/t/how-to-upload-large-files-size-to-your-submission/2304).

## How to submit a trained model!

To make a submission, you will have to create a **private** repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).

You will have to add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

Then you can create a submission by making a _tag push_ to your repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).
**Any tag push (where the tag name begins with "submission-") to your private repository is considered as a submission**  
Then you can add the correct git remote, and finally submit by doing :

```
cd music-demixing-challenge-starter-kit
# Add AIcrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com:<YOUR_AICROWD_USER_NAME>/music-demixing-challenge-starter-kit.git
git push aicrowd master
```

```
# Create a tag for your submission and push
git tag -am "submission-v0.1" submission-v0.1
git push aicrowd master
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change,
# then pushing a new tag will **not** trigger a new evaluation.
```

You now should be able to see the details of your submission at :
[gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/music-demixing-challenge-starter-kit/issues](https://gitlab.aicrowd.com//<YOUR_AICROWD_USER_NAME>/music-demixing-challenge-starter-kit/issues)

**NOTE**: Remember to update your username instead of `<YOUR_AICROWD_USER_NAME>` above :wink:

After completing above steps, you should start seeing something like below to take shape in your Repository -> Issues page:
![](https://i.imgur.com/6jzlIdx.png)

and if everything works out correctly, then you should be able to see the final scores like this :
![](https://i.imgur.com/DXnUKHP.png)

### Other helpful files

👉 [RUNTIME.md](/docs/RUNTIME.md)
