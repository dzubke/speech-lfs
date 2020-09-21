# standard libs
import argparse
import glob
from multiprocessing import Pool
import os
import random
# third-party libs
# project libs


def remove_random(data_dir:str, file_ext:str, percent_removed:float):
    """
    Removes a random percentage of files in a directory with a given file extension.
    Args:
        data_dir: Directory where files will be removed
        file_ext: File extension of the files to be removed. Files without that extension will be ignored. 
        percent_removed: Decimal percent of files with given extension that will bre removed.
    Output:
        None
    """
    
    NUM_PROC = 200

    # create the file paths
    pattern = "*." + file_ext
    regex = os.path.join(data_dir, pattern)    
    file_paths = glob.glob(regex)
    
    # randomize the files
    random.shuffle(file_paths)

    # create the subset to remove
    assert 0 < percent_removed < 1.0
    num_to_remove = int(len(file_paths) * percent_removed)
    files_to_remove = file_paths[:num_to_remove]

    # remove the files
    # single process approach
    #for file_path in files_to_remove:
    #    os.remove(file_path)
    
    # mutlit-process approach
    pool = Pool(processes=NUM_PROC)
    pool.imap_unordered(os.remove, files_to_remove, chunksize=100)
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Removes a random percentage of files in a directory with a given file extension."
    )
    parser.add_argument(
        "--data-dir", type=str, help="Directory where files will be removed"
    )
    parser.add_argument(
        "--ext", type=str, help="File extension of the files to be removed."
    )
    parser.add_argument(
        "--percent-removed", type=float, 
        help="Decimal percent of files with given extension that will bre removed."
    )
    args = parser.parse_args()

    remove_random(args.data_dir, args.ext, args.percent_removed)
