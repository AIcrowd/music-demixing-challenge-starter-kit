import os, sys
sys.path.append(os.path.dirname(os.path.realpath(os.getcwd())))
sys.path.append(os.path.realpath(os.getcwd()))

import musdb

if __name__ == "__main__":
    dataset = musdb.DB(root="data/MUSDB18/", download=True)
    print("Download complete.")