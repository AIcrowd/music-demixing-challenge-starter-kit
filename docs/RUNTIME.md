## Adding your runtime

This repository is a valid submission (and submission structure). 
You can simply add your dependencies on top of this repository.

Few of the most common ways are as follows:

* `environment.yml` -- The _optional_ Anaconda environment specification. 
    As you add new requirements you can export your `conda` environment to this file!
    ```
    conda env export --no-build > environment.yml
    ```

    * **Create your new conda environment**

        ```sh
        conda create --name music_demixing_challenge
        conda activate music_demixing_challenge
        ```

  * **Your code specific dependencies**
    ```sh
    conda install <your-package>
    ```  


* `requirements.txt` -- The `pip3` packages used by your inference code. **Note that dependencies specified by `environment.yml` take precedence over `requirements.txt`.** As you add new pip3 packages to your inference procedure either manually add them to `requirements.txt` or if your software runtime is simple, perform:
    ```
    # Put ALL of the current pip3 packages on your system in the submission
    >> pip3 freeze >> requirements.txt
    >> cat requirementst.txt
    aicrowd_api
    coloredlogs
    matplotlib
    pandas
    [...]
    ```

* `apt.txt` -- The Debian packages (via aptitude) used by your inference code!

These files are used to construct your **AIcrowd submission docker containers** in which your code will run. 

In case you are advanced user, you can check other methods to specify the runtime [here](https://discourse.aicrowd.com/t/how-to-specify-runtime-environment-for-your-submission/2274), which includes adding your own `Dockerfile` directly.

----

👋 In case you have any doubts or need help, you can reach out to us via Challenge [Discussions](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021/discussion) or [Discord](https://discord.gg/fNRrSvZkry).
