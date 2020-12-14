# Copyright 2020 Speak Labs

"""
This script filters the a training.json files further
based on the number of files and certain constraints. This is mostly
used for the speak-train dataset.
"""

# standard libs
import argparse
from collections import defaultdict
import csv
import json
import os
import random
from typing import List
# third-party libs
import yaml
# project libs
from speech.utils.io import read_data_json
from speech.utils.data_helpers import check_disjoint_filter, check_update_contraints, get_dataset_ids 
from speech.utils.data_helpers import path_to_id, process_text

def filter_speak_train(
    full_json_path:str, 
    metadata_path:str, 
    filter_json_path:str,
    dataset_size: int, 
    constraints:dict,
    disjoint_datasets: dict)->None:
    """
    This script filters the dataset in `full_json_path` and write the new dataset to `filter_json_path`.
    The constraints on the filtered dataset are:
        - utterances per speaker, lesson, and line cannot exceed the decimal values 
            as a fraction of the `dataset_size`. 
            Older config files have an absolute value on the `max_speaker_count`
        - the utterances are not also included in the datasets specified in `excluded_datasets`

    Args:
        full_json_path (str): path to the source json file that that the output will filter from
        metadata_path (str): path to the tsv file that includes metadata on each recording, 
            like the speaker_id
        filter_json_path (str): path to the filtered, written json file
        dataset_size (int): number of utterances included in the output dataset
        constraints (dict): dict of constraints on the number of utterances per speaker, lesson, 
            and line expressed as decimal fractions of the total dataset.
        disjoint_datasets (Dict[Tuple[str],str]): dict whose keys are a tuple of the ids that will be disjoint
            and whose values are the datasets paths whose examples will be disjiont from the output
    Returns:
        None, only files written.
    """

    # re-calculate the constraints as integer counts based on the `dataset_size`
    constraints = {name: int(constraints[name] * dataset_size) for name in constraints.keys()}
    print("constraints: ", constraints)

    # constraint_names will help to ensure the dict keys created later are consistent.
    constraint_names = list(constraints.keys())

    # read and shuffle the full dataset and convert to iterator to save memory
    full_dataset = read_data_json(full_json_path)
    random.shuffle(full_dataset)
    full_dataset = iter(full_dataset)

    # create a mapping from record_id to lesson, line, and speaker ids
    with open(metadata_path, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        header = next(tsv_reader)
        # header: id, text, lessonId, lineId, uid(speaker_id), redWords_score, date
        print("header: ", header)
        # this assert helps to ensure the row indexing below is correct
        assert len(header) == 7, \
            f"metadata header is not expected length. Expected 7, got {len(header)}."
        # mapping from record_id to other ids like lesson, speaker, and line
        record_ids_map = dict()
        for row in tsv_reader:
            tar_sentence = process_text(row[1])
            record_ids_map[row[0]] = {
                    "record": row[0],                    # adding record for disjoint_check
                    constraint_names[0]: row[2],        # lesson
                    constraint_names[1]: tar_sentence,  # using target_sentence instead of lineId
                    constraint_names[2]: row[4]         # speaker
            }


    # create a defaultdict with set values for each disjoint-id name
    disjoint_id_sets = defaultdict(set)

    for dj_data_path, dj_names in disjoint_datasets.items():
        # get all the record_ids in the dataset
        record_ids = get_dataset_ids(dj_data_path)
        # loop through the disjoint-id-names in the key-tuple
        for dj_name in dj_names:
            for record_id in record_ids:
                # add the id to the relevant id-set
                disjoint_id_sets[dj_name].add(record_ids_map[record_id][dj_name])
    
    print("all disjoint names: ", disjoint_id_sets.keys())


    # id_counter keeps track of the counts for each speaker, lesson, and line ids
    id_counter = {name: dict() for name in constraint_names}

    examples_written = 0
    # loop until the number of examples in dataset_size has been written
    with open(filter_json_path, 'w') as fid:
        while examples_written < dataset_size:
            if examples_written != 0 and examples_written % 100000 == 0:
                print(f"{examples_written} examples written")
            try:
                example = next(full_dataset)
            except StopIteration:
                print(f"Stop encountered {examples_written} examples written")
                break
                
            record_id = path_to_id(example['audio'])
            # check if the ids associated with the record_id are not included in the disjoint_datasets
            pass_filter = check_disjoint_filter(record_id, disjoint_id_sets, record_ids_map)
            if pass_filter:
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="filters a training dataset")
    parser.add_argument("--config", type=str,
        help="path to preprocessing config.")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file)

    print("config: ", config)
    if config['dataset_name'].lower() == "speaktrain":
        filter_speak_train(
            config['full_json_path'],
            config['metadata_tsv_path'],
            config['filter_json_path'],
            config['dataset_size'],
            config['constraints'],
            config['disjoint_datasets'] 
        ) 
