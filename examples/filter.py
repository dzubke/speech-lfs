# This script is use to filter the training.json files further
# based on the number of files and certain constraints. This is mostly
# used by the speak-train dataset.


# standard libs
import argparse
import csv
import json
import os
import random
import time
# third-party libs
import yaml
# project libs
from speech.utils.io import read_data_json

def filter_speak_train(full_json_path:str, 
                        metadata_path:str, 
                        filter_json_path:str,
                        dataset_size: int, 
                        max_speaker_count:int):

    print("max_speaker_count", max_speaker_count)

    # read and shuffle the full dataset and convert to iterator
    full_dataset = read_data_json(full_json_path)
    random.shuffle(full_dataset)
    full_dataset = iter(full_dataset)

    # create a mapping from record_id to speaker_id from the metadata.tsv
    with open(metadata_path, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        # header: id, text, lessonId, lineId, uid(speaker_id), date
        header = next(tsv_reader)
        record_speaker_map = {row[0]: row[4] for row in tsv_reader}
            
    # dict and int to count number of speakers and examples written, respectively
    speaker_counter = dict()
    examples_written = 0
    start_time = time.time()
    # loop until the number of examples in dataset_size has been written
    with open(filter_json_path, 'w') as fid:
        while examples_written < dataset_size:
            example = next(full_dataset)
            record_id = os.path.basename(
                os.path.splitext(example['audio'])[0]
            )
            speaker_id = record_speaker_map[record_id]
            speaker_count = speaker_counter.get(speaker_id, 0)
            # if criterion is met, write the example
            if speaker_count < max_speaker_count:
                json.dump(example, fid)
                fid.write("\n")
                # increment counters
                examples_written += 1
                speaker_counter[speaker_id] = speaker_count + 1




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="filters a training dataset")
    parser.add_argument("--config", type=str,
        help="path to preprocessing config.")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)

    if config['dataset_name'].lower() == "speaktrain":
        filter_speak_train(config['full_json_path'],
                            config['metadata_tsv_path'],
                            config['filter_json_path'],
                            config['dataset_size'],
                            config['max_speaker_count']  
        ) 
