
### 1. Download

1. [Full dataset](https://sigsep.github.io/datasets/musdb.html)
    - The entire dataset is hosted on Zenodo and requires that users request access.
    - The tracks can only be used for academic purposes. 
    - They manually check requests. 
    - After your request is accepted, then you can download the full dataset!

2. or Sample Dataset
    - Download sample version of MUSDB18 which includes 7s excerpts
    
        ```shell script
        python dl_musdb18_samples.py
        ```
    - or using this script
 
        ```python
        import musdb
        from lasaft.utils.functions import mkdir_if_not_exists
        mkdir_if_not_exists('etc')
        musdb.DB(root='etc/musdb18_dev', download=True)
        ```

### 2. Generate wave files

- run this!

    ```shell
    musdbconvert <your_DIR> <target_DIR> 
    ```
  
  - for example, ```musdbconvert etc\musdb18_dev etc\musdb18_dev_wav```

- musdbconvert is automatically installed if you have installed musdb with:

    ```shell
    pip install musdb
    ```
