# Copyright 2020 Speak Labs

"""
This script filters the a training.json files further
based on the number of files and certain constraints. This is mostly
used for the speak-train dataset.
"""

# standard libs
import argparse
import csv
import json
import os
import random
from typing import List
# third-party libs
import yaml
# project libs
from speech.utils.io import read_data_json

def filter_speak_train(
    full_json_path:str, 
    metadata_path:str, 
    filter_json_path:str,
    dataset_size: int, 
    constraints:dict,
    excluded_datasets: List[str])->None:
    """
    This script filters the dataset in `full_json_path` and write the new dataset to `filter_json_path`.
    The constraints on the filtered dataset are:
        - utterances per speaker, lesson, and line cannot exceed the decimal values 
            as a fraction of the `dataset_size`. 
            Older config files have an absolute value on the `max_speaker_count`
        - the utterances are not also included in the datasets specified in `excluded_datasets`

    Args:
        full_json_path (str): path to the full dataset file
        metadata_path (str): path to the tsv file that includes metadata on each recording, 
            like the speaker_id
        filter_json_path (str): path to the filtered, written json file
        dataset_size (int): number of utterances included in the output dataset
        constraints (dict): dict of constraints on the number of utterances per speaker, lesson, 
            and line expressed as decimal fractions of the total dataset.
        excluded_datasets (List[str]): list of paths to datasets whose samples will be excluded from 
            the output dataset
    Returns:
        None, only files written.
    """

    # re-calculate the constraints as integer counts based on the `dataset_size`
    constraints = {name: int(constraints[name] * dataset_size) for name in constraints.keys()}
    # constraint_names will help to ensure the dict keys created later are consistent.
    constraint_names = ['lesson', 'line', 'speaker']
    assert constraint_names == list(constraints.keys()), \
        f"names of constraints do not match: {constraint_names}, {list(constraint.keys())}"
    print("constraints: ", constraints)

    # read and shuffle the full dataset and convert to iterator to save memory
    full_dataset = read_data_json(full_json_path)
    random.shuffle(full_dataset)
    full_dataset = iter(full_dataset)

    # create a mapping from record_id to lesson, line, and speaker ids
    with open(metadata_path, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        # header: id, text, lessonId, lineId, uid(speaker_id), date
        header = next(tsv_reader)
        record_ids_map = {
            row[0]: {
                constraint_names[0]: row[2],   # lesson
                constraint_names[1]: row[3],    # line
                constraint_names[2]: row[4]     # speaker
            }
        for row in tsv_reader}

    # create a set of all the record_id's in the excluded datasets that will not
    ## be included in the filtered dataset
    def _extract_id(record_path:str)->str:
        #returns the basename of the path without the extension
        return os.path.basename(
            os.path.splitext(record_path)[0]
        )
    
    excluded_record_ids = set()
    for ex_dataset_path in excluded_datasets:
        ex_dataset = read_data_json(ex_dataset_path)
        # extracts the record_ids from the excluded datasets
        record_ids = [_extract_id(example['audio']) for example in ex_dataset]
        excluded_record_ids.update(record_ids)    
            
    # id_counter keeps track of the counts for each speaker, lesson, and line ids
    id_counter = {
        constraint_names[0]: dict(),    # lesson
        constraint_names[1]: dict(),    # line
        constraint_names[2]: dict()     # speaker
    }

    examples_written = 0
    # loop until the number of examples in dataset_size has been written
    with open(filter_json_path, 'w') as fid:
        while examples_written < dataset_size:
            if examples_written % 50000 == 0 :
                print(f"{examples_written} examples written")
            example = next(full_dataset)
            record_id = _extract_id(example['audio'])
            # check if record_id is already in an excluded dataset
            if record_id not in excluded_record_ids:
                # check if the record_id pass the speaker, line, lesson constraints
                pass_constraint = check_update_contraints(
                    record_id, 
                    record_ids_map,
                    id_counter, 
                    constraints
                )
                if pass_constraint:
                    json.dump(example, fid)
                    fid.write("\n")
                    # increment counters
                    examples_written += 1

def check_update_contraints(record_id:int, 
                            record_ids_map:dict,
                            id_counter:dict, 
                            constraints:dict)->bool:
    """Checks if the counts for the `record_id` is less than the constraints. If the
    count is less (constraint passes) increment the `id_counter`
  
    Args:
        record_id (int): id of the record
        record_id_map (dict): dict that maps record_ids to speaker, lesson, and line ids
        id_counter (dict): dict of counts of speaker, lesson, and line ids
        constraints (dict): dict of 3 ints specifying the max number of utterances
            per speaker, line, and lesson
    Returns:
        bool: true if the count of utterances per speaker, lesson, and line are all
            below the max value in `constraints`
    """
    pass_constraint = True
    # constraint_names = ['lesson', 'line', 'speaker']
    constraint_names = list(constraints.keys())

    for name in constraint_names:
        constraint_id = record_ids_map[record_id][name]
        count = id_counter[name].get(constraint_id, 0)
        if count > constraints[name]:
            pass_constraint = False
            break
    
    # if `record_id` passes the constraint, update the `id_counter`
    if pass_constraint:
        for name in constraint_names:
            constraint_id = record_ids_map[record_id][name]
            id_counter[name][constraint_id] = id_counter[name].get(constraint_id, 0) + 1

    return pass_constraint



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="filters a training dataset")
    parser.add_argument("--config", type=str,
        help="path to preprocessing config.")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)

    if config['dataset_name'].lower() == "speaktrain":
        filter_speak_train(
            config['full_json_path'],
            config['metadata_tsv_path'],
            config['filter_json_path'],
            config['dataset_size'],
            config['constraints'],
            config['excluded_datasets'] 
        ) 
