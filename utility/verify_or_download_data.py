import os, sys
sys.path.append(os.path.dirname(os.path.realpath(os.getcwd())))
sys.path.append(os.path.realpath(os.getcwd()))

from core.dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset(local_path='data/part1', s3_path='s3://airborne-obj-detection-challenge-training/part1/')
    dataset.add(local_path='data/part2', s3_path='s3://airborne-obj-detection-challenge-training/part2/')
    dataset.add(local_path='data/part3', s3_path='s3://airborne-obj-detection-challenge-training/part3/')

    i = 1
    all_flights = dataset.get_flight_ids()
    for flight_id in all_flights:
        print("Downloading Flight#%s (%s/%s)..." % (flight_id, i, len(all_flights)))
        dataset.get_flight_by_id(flight_id).download()
        i += 1

    print("Download complete.")
