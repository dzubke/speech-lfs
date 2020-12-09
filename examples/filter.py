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
from speech.utils.data_helpers import check_disjoint_filter, check_update_contraints, path_to_id
from speech.utils.data_helpers import process_text

def filter_speak_train(
    full_json_path:str, 
    metadata_path:str, 
    filter_json_path:str,
    dataset_size: int, 
    constraints:dict,
    disjoint_id_names: List[str],
    disjoint_datasets: List[str])->None:
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
        distjoint_id_names (List[str]): list of names of ids that specify the ids that will be disjoint
        disjoint_datasets (List[str]): list of paths to datasets whose samples will be disjiont from 
            the output dataset along the `disjoint_ids`
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
        print("header: ", header)
        # this assert helps to ensure the row indexing below is correct
        assert len(header) == 7, \
            f"metadata header is not expected length. Expected 7, got {len(header)}."
        # header: id, text, lessonId, lineId, uid(speaker_id), redWords_score, date
        for row in tsv_reader:
            tar_sentence = process_text(row[1])
            record_ids_map.update({
                row[0]: {
                    "record": row[0],                    # adding record for disjoint_check
                    constraint_names[0]: row[2],        # lesson
                    constraint_names[1]: tar_sentence,  # using target_sentence instead of lineId
                    constraint_names[2]: row[4]         # speaker
                }
            })

    # create a dict of sets of all the ids in the disjoint datasets that will not
    # be included in the filtered dataset
    disjoint_id_sets = {name: set() for name in disjoint_id_names}
    for disj_dataset_path in disjoint_datasets:
        disj_dataset = read_data_json(disj_dataset_path)
        # extracts the record_ids from the excluded datasets
        record_ids = [path_to_id(example['audio']) for example in disj_dataset]
        # loop through each record id
        for record_id in record_ids:
            # loop through each id_name and update the disjoint_id_sets
            for disjoint_id_name, disjoint_id_set in disjoint_id_sets.items():
                disjoint_id_set.add(record_ids_map[record_id][disjoint_id_name])

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
            if examples_written % 250 == 0 :
                print(f"{examples_written} examples written")
            example = next(full_dataset)
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

    if config['dataset_name'].lower() == "speaktrain":
        filter_speak_train(
            config['full_json_path'],
            config['metadata_tsv_path'],
            config['filter_json_path'],
            config['dataset_size'],
            config['constraints'],
            config['disjoint_ids'],
            config['disjoint_datasets'] 
        ) 
