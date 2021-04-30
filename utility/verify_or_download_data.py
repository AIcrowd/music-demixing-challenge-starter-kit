import os, sys
import requests
import zipfile

sys.path.append(os.path.dirname(os.path.realpath(os.getcwd())))
sys.path.append(os.path.realpath(os.getcwd()))

DATASET_FILE_NAME = 'download.zip'
DATASET_DOWNLOAD_URL = 'https://zenodo.org/record/3270814/files/MUSDB18-7-WAV.zip?download=1'
DATASET_FULL_DOWNLOAD_URL = 'https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1'

def download_dataset(full=False):
    dn_url = DATASET_DOWNLOAD_URL
    if full:
        dn_url = DATASET_FULL_DOWNLOAD_URL
    r = requests.get(dn_url, stream=True)
    with open(DATASET_FILE_NAME, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=256):
            fd.write(chunk)

def unzip_dataset():
    with zipfile.ZipFile(DATASET_FILE_NAME, 'r') as zip_ref:
        zip_ref.extractall('data/')

def cleanup():
    os.remove(DATASET_FILE_NAME)

def verify_dataset():
    assert os.path.isdir("data/train") and os.path.isdir("data/test"), "Dataset folder not found"
    assert os.path.isdir("data/train/Hollow Ground - Left Blind"), "Random song check in training folder failed"
    assert os.path.isdir("data/test/Louis Cressy Band - Good Time"), "Random song check in testing folder failed"

def move_to_git_root():
    if not os.path.exists(os.path.join(os.getcwd(), ".git")):
        os.chdir("..")
    assert os.path.exists(os.path.join(os.getcwd(), ".git")), "Unable to reach to repository root"

if __name__ == "__main__":
    move_to_git_root()
    try:
        verify_dataset()
    except AssertionError:
        print("Dataset not found...")
        option = input("Download full dataset (y/Y) or 7s dataset (n/N)? ")
        if option.lower() == 'y':
            print("Downloading full dataset...")
            download_dataset(full=True)
        else:
            print("Downloading 7s dataset...")
            download_dataset()
        print("Unzipping the dataset...")
        unzip_dataset()
        print("Cleaning up...")
        #cleanup()
        print("Verifying the dataset...")
        verify_dataset()
        print("Done.")
