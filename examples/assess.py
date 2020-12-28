"""
this script is meant to assess a dataset along a variety of measures
author: Dustin Zubke
license: MIT
"""
# standard libary
import argparse
from collections import Counter, OrderedDict
from functools import partial
import csv
import os
import re
from typing import List
# third party libraries
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import pandas as pd
# project libraries
from speech.dataset_info import AllDatasets, TatoebaDataset
from speech.utils.data_helpers import get_record_id_map, get_dataset_ids, path_to_id
from speech.utils.data_helpers import print_symmetric_table, process_text
from speech.utils.io import read_data_json, write_pickle


def assess_commonvoice(validated_path:str, max_occurance:int):

    val_df = pd.read_csv(validated_path, delimiter='\t',encoding='utf-8')
    print(f"there are {val_df.shape[0]} entries/rows in the dataset")
    accents=["us", "canada"]    
    # 231011 rows with accents "us" and "canada", 206653 with us and 24358 with canada 
    val_df = val_df[val_df.accent.isin(accents)]
    print(f"there are {val_df.shape[0]} entries with accents {accents}")
    # create vote_diff column to sort the sentences upon
    val_df["vote_diff"] = val_df.up_votes - val_df.down_votes
    # remove punctiation and lower the case in sentence
    val_df['sentence']=val_df['sentence'].str.replace('[^\w\s]','').str.lower() 
    # sorts by the number of unique utterances in descending order
    val_df.sentence.value_counts(sort=True, ascending=False)
    # histogram bins
    #pd.cut(val_df.sentence.value_counts(sort=True, ascending=False),bin_range).value_counts().sort_index() 
    # dictionary of frequency counts
    count_dict=val_df.sentence.value_counts(sort=True, ascending=False).to_dict() 
    # filters so utterances only have at most max_occurances
    # keeps utterances with highest difference between up_votes and down_votes
    val_df, drop_row_count = filter_by_count(val_df, count_dict, max_occurance)
    print(f"number of rows dropped: {drop_row_count}")
    dirname = os.path.dirname(validated_path)
    write_path = os.path.join(dirname, f"validated-{max_occurance}-maxrepeat.tsv")
    if os.path.exists(write_path):
        print(f"file: {write_path} already exists.")
        print("Would you like to rewrite it? y/n")
        answer = input()
        if answer in ["Y", "y"]:
            val_df.to_csv(write_path, sep="\t", index=False)
            print(f"file: {write_path} successfully saved")
        else: 
            print("file has not be overwritten. No new file saved")
    else:
        val_df.to_csv(write_path, sep="\t", index=False)
        print(f"file: {write_path} successfully saved")



def filter_by_count(in_df:pd.DataFrame, count_dict:dict, filter_value:int):
    """
    filters the dataframe so that seteneces that occur more frequently than
    the fitler_value are reduced to a nubmer of occurances equal to the filter value,
    sentences to be filters will be done based on the difference between the up_votes and down_votes
    """
    drop_row_count = 0 
    for sentence, count in count_dict.items():
        if count > filter_value:
            # selecting rows that equal sentence
            # then sorting that subset by the vote_diff value in descending order
            # then taking the indicies of the rows after the first # of filter_values
            drop_index = in_df[in_df.sentence.eq(sentence)]\
            .sort_values("vote_diff", ascending=False)\
            .iloc[filter_value:,:].index
            
            drop_row_count += len(drop_index)
            # dropping the rows in drop_index
            in_df = in_df.drop(index=drop_index)
    return in_df, drop_row_count



def assess_iphone_models(save_path:str)->None:
    """This function seeks to identify the distribution of iphone models across a random sample of 
    Speak's userbase. A historgram will be created of the number of users on each iphone model. 
    Args:
        save_path (str): path where iphone count will be saved as pickle
    """
    PROJECT_ID = 'speak-v2-2a1f1'
    QUERY_LIMIT = 10000
    
    # verify and set the credientials
    CREDENTIAL_PATH = "/home/dzubke/awni_speech/speak-v2-2a1f1-d8fc553a3437.json"
    # CREDENTIAL_PATH = "/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speak-v2-2a1f1-d8fc553a3437.json"
    # set the enviroment variable that `firebase_admin.credentials` will use
    os.putenv("GOOGLE_APPLICATION_CREDENTIALS", CREDENTIAL_PATH)

    # initialize the credentials and firebase db client
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {'projectId': PROJECT_ID})
    db = firestore.client()

    rec_ref = db.collection(u'recordings')
    iphone_model_count = Counter()
    n_iphone_models = 100000

    while sum(iphone_model_count.values()) < n_iphone_models:
        print("inside while loop")
        next_query = rec_ref.order_by(u'id').limit(QUERY_LIMIT)
        for doc in next_query.stream():
            doc = doc.to_dict()
            # only select dates in 2020
            rec_date = doc.get('info', {}).get('date', None)
            if isinstance(rec_date, str):
                if rec_date.startswith('2020'):
                    # get the iphone model
                    iphone_model = doc.get('user', {}).get('deviceModelIdentifier', None)
                    if iphone_model is not None:
                        # iphone_model has the formate 'iPad8,2', so splitting off second half
                        iphone_model = iphone_model.split(',')[0]
                        iphone_model_count[iphone_model] += 1 

    #iphone_model_count = dict(iphone_model_count)
    write_pickle(save_path, iphone_model_count)

    # plot the iphone model counts
    model_names, model_counts = list(zip(*iphone_model_count.most_common()))
    plt.plot(model_names, model_counts)
    plt.xticks(model_names, model_names, rotation=45)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(model_names, model_counts)
    plt.xticks(model_names, model_names, rotation=45)
    total = sum(model_counts)
    
    # plot the aggregate and percent of total values on both axes
    def _agg2percent_forward(x, total):
        return x/total

    def _agg2percent_backward(x, total):
        return x*total

    # create the forward and backward transforms for the axis
    forward_transform = partial(_agg2percent_forward, total=total)
    backward_transform = partial(_agg2percent_backward, total=total)
    # create the secondary axis
    secaxy = ax.secondary_yaxis('right', functions=(forward_transform,
                                                    backward_transform))

    # add the plot labels for each axis
    ax.set_ylabel("Device model count")
    secaxy.set_ylabel("Percent of total device count")
    plt.xlabel("Device names")



def assess_speak_train(dataset_paths: List[str], 
                        metadata_path:str, 
                        out_dir:str, 
                        use_json:bool=True)->None:
    """This function creates counts of the speaker, lesson, and line ids in a speak training dataset
    Args:
        dataset_path (str): path to speak training.json dataset
        metadata_path (str): path to tsv file that contains speaker, line, and lesson ids 
        out_dir (str): directory where plots and txt files will be saved
        use_json (bool): if true, the data will be read from a training.json file
    Returns:
        None
    """


    def _increment_key(in_dict, key): 
        in_dict[key] = in_dict.get(key, 0) + 1


    def _plot_count(ax, count_dict:dict, label:str):
        ax.plot(range(len(count_dict.values())), sorted(list(count_dict.values()), reverse=True))
        ax.set_title(label)
        ax.set_xlabel(f"unique {label}")
        ax.set_ylabel(f"utterance per {label}")
        ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));
        ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));
        plt.tight_layout()

    def reformat_large_tick_values(tick_val, pos):
        """
        Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and 
        also appropriately turns 4000 into 4K (no zero after the decimal).
        taken from: https://dfrieds.com/data-visualizations/how-format-large-tick-values.html
        """
        if tick_val >= 1000000000:
            val = round(tick_val/1000000000, 1)
            new_tick_format = '{:}B'.format(val)
        elif tick_val >= 1000000:
            val = round(tick_val/1000000, 1)
            new_tick_format = '{:}M'.format(val)
        elif tick_val >= 1000:
            val = round(tick_val/1000, 1)
            new_tick_format = '{:}K'.format(val)
        elif tick_val < 1000:
            new_tick_format = round(tick_val, 1)
        else:
            new_tick_format = tick_val
        
        return str(new_tick_format)


    def _print_stats(count_dict:dict):
        values = list(count_dict.values())
        mean = round(np.mean(values), 2)
        std = round(np.std(values), 2)
        max_val = round(max(values), 2)
        min_val = round(min(values), 2)
        print(f"mean: {mean}, std: {std}, max: {max_val}, min: {min_val}, total_unique: {len(count_dict)}")
        print(f"sample of 5 values: {list(count_dict.keys())[0:5]}")
    
    # this will read the data from a metadata.tsv file
    if not use_json:
        # count dictionaries for the lesssons, lines, and users (speakers)
        lesson_dict, line_dict, user_dict, target_dict = {}, {}, {}, {}
        # create count_dicts for each
        with open(metadata_path, 'r') as tsv_file: 
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            header = next(tsv_reader) 
            print(header) 
            for row in tsv_reader: 
                _increment_key(lesson_dict, row[2]) 
                _increment_key(line_dict, row[3]) 
                _increment_key(user_dict, row[4]) 
                _increment_key(target_dict, process_text(row[1])) 

        # put the labels and count_dicts in list of the for-loop
        constraint_names = ['lesson', 'line', 'speaker', 'target_sent']
        counter = {
            "lesson": lesson_dict, 
            "line": line_dict, 
            "speaker": user_dict,
            "target_sent": target_dict
        }

    # reading from a training.json file supported by a metadata.tsv file
    if use_json:
        # create mapping from record_id to speaker, line, and lesson ids
        rec_ids_map = dict()
        constraint_names = ['lesson', 'line', 'speaker', 'target_sent']
        counter = {name: dict() for name in constraint_names}
        with open(metadata_path, 'r') as tsv_file: 
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            # header: id, text, lessonId, lineId, uid(speaker_id), date
            header = next(tsv_reader)
            rec_ids_map = dict()
            for row in tsv_reader:
                rec_ids_map[row[0]]= {
                        constraint_names[0]: row[2],   # lesson
                        constraint_names[1]: row[3],    # line
                        constraint_names[2]: row[4],    # speaker
                        constraint_names[3]: process_text(row[1]),  # target-sentence
                        "date": row[6]                  # date
                }

        total_date_counter = dict()
        # `unq_date_sets` keep track of the unique ids
        unq_date_counter = {name: dict() for name in constraint_names}
        # iterate through the datasets
        for dataset_path in dataset_paths:
            dataset = read_data_json(dataset_path)
            print(f"dataset {path_to_id(dataset_path)} size is: {len(dataset)}")

            # iterate through the exmaples in the dataset
            for xmpl in dataset:
                rec_id = path_to_id(xmpl['audio'])
                date =  rec_ids_map[rec_id]['date']
                # date has format 2020-09-10T04:24:03.073Z, so splitting
                # and joining by '-' using the first two element will be `2020-09`
                yyyy_mm_date = '-'.join(date.split('-')[:2])
                _increment_key(total_date_counter, yyyy_mm_date)

                # iterate through the constraints and update the id counters
                for name in constraint_names:
                    constraint_id = rec_ids_map[rec_id][name]
                    _increment_key(counter[name], constraint_id)
                    update_unq_date_counter(
                        unq_date_counter, 
                        name, 
                        constraint_id,
                        yyyy_mm_date
                    )

                
    
    # create the plots
    fig, axs = plt.subplots(1,len(constraint_names))
    fig.set_size_inches(8, 6)

    # plot and calculate stats of the count_dicts
    for ax, name in zip(axs, constraint_names):
        _plot_count(ax, counter[name], name)
        print(f"{name} stats")
        _print_stats(counter[name])
        print()
    
    # ensures the directory of `out_dir` exists
    os.makedirs(out_dir, exist_ok=dir)
    out_path = os.path.join(out_dir, os.path.basename(out_dir))
    print("out_path: ", out_path)
    plt.savefig(out_path + "_count_plot.png")
    plt.close()

    # plot the total_date histogram
    fig, ax = plt.subplots(1,1)
    dates = sorted(total_date_counter.keys())
    date_counts = [total_date_counter[date] for date in dates]
    ax.plot(range(len(date_counts)), date_counts)
    plt.xticks(range(len(date_counts)), dates, rotation=60)
    #ax.set_title(label)
    #ax.set_xlabel(f"unique {label}")
    #ax.set_ylabel(f"utterance per {label}")
    #ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));
    plt.tight_layout()

    plt.savefig(out_path + "_date_count.png")
    plt.close()

    # plot the unique ids
    for name in constraint_names:
        fig, ax = plt.subplots(1,1)
        date_counts = []
        dates = sorted(unq_date_counter[name].keys())
        total_count = sum([unq_date_counter[name][date]['count'] for date in dates])
        cumulative_count = 0
        for date in dates:
            cumulative_count += unq_date_counter[name][date]['count'] 
            date_counts.append(round(cumulative_count/total_count, 2))
        
        ax.plot(range(len(date_counts)), date_counts)
        plt.xticks(range(len(date_counts)), dates, rotation=60)
        ax.set_title(name)
        ax.set_xlabel(f"Date")
        ax.set_ylabel(f"% of total unique ID's")
        #ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));
        #ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values));
        plt.tight_layout()

        plt.savefig(out_path + f"_unq_cum_date_{name}.png")
        plt.close()


    # sort the lesson_ids and line_ids and write to txt file
    for name in counter:
        sorted_ids = sorted(list(counter[name].keys()))

        with open(f"{out_path}_{name}.txt", 'w') as fid:
            for ids in sorted_ids:
                fid.write(ids+"\n")


    #print("unique lessons")
    #print(sorted(list(lesson_dict.keys()))[:200])
    #print(f"number of unique lessons: {len(set(lesson_dict.keys()))}")


def dataset_stats(dataset_path:str)->None:
    """This function prints a variety of stats (like mean and std-dev) for the input dataset

    Args:
        dataset_path (str): path to the dataset
    """

    dataset = read_data_json(dataset_path)

    data_features = {
        "target_len": [len(xmpl['text']) for xmpl in dataset],
        "audio_dur": [xmpl['duration'] for xmpl in dataset]
    }

    stat_functions = {
        "mean": np.mean,
        "stddev": np.std,
    }

    print(f"stats for dataset: {os.path.basename(dataset_path)}")
    for data_name, data in data_features.items():
        for stat_name, stat_fn in stat_functions.items():
            print(f"\t{stat_name} of {data_name} is: {round(stat_fn(data), 3)}")
        print()


def dataset_overlap(dataset_list: str, 
                    metadata_path: str,
                    overlap_key: str)->None:
    """This function assess the overlap between two datasets by the `overlap_key`. 
    Two metrics are calcualted: 
        1) coutn of unique overlap_keys / total unique overlap_keys
        2) count of total overlaping keys / total records

    Args:
        dataset_list (List[str]): list of dataset paths to compare
        metadata_path (str): path to metadata tsv file
        overlap_key (str): key to assess overlap (like speaker_id or target-sentence)

    Returns:
        None
    """

    print(f"assessing overlap based on key: {overlap_key}")

    record_id_map = get_record_id_map(metadata_path)

    # creates a shorter, pretty name of the dataset
    def pretty_data_name(data_name):
        """This function makes the data name shorter and easier to read
        """
        data_name = os.path.basename(data_name)             # remove the path directories
        data_name = os.path.splitext(data_name)[0]          # removes extension
        data_name = data_name.replace("speak-", "")         # remove 'speak-'
        data_name = data_name.replace("data_trim", "7M")    # changes name for 7M records
        data_name = data_name.replace("eval2_data", "eval2-v1") # change the eval2-v1 name
        data_name = data_name.replace("_data", "")          # removes _data from v4 and v5
        data_name = re.sub(r'_2020-..-..', '',data_name)    # removes date
        return data_name

    data_dict = {
        pretty_data_name(datapath): get_dataset_ids(datapath)
        for datapath in dataset_list
    }

    # check the record_id_map contains all of the records in data1 and data2
    rec_map_set = set(record_id_map.keys())

    for data_name, data_ids in data_dict.items():
        assert data_ids <= rec_map_set, \
            f"{data_name} ids not in record_id_map:\n {data_ids.difference(rec_map_set)}"

    # delete to save memory
    del rec_map_set

    data_keyid_lists = dict()
    for data_name, rec_ids in data_dict.items():
        data_keyid_lists[data_name] = [
            record_id_map[rec_id][overlap_key] for rec_id in rec_ids
        ]

    data_keyid_sets = {
        data_name: set(key_ids)
        for data_name, key_ids in data_keyid_lists.items()
    }
    data_keyid_counters ={
        data_name: Counter(key_ids)
        for data_name, key_ids in data_keyid_lists.items()
    }
    # reference dataset to be analyzed
    unq_output = dict()
    for ref_name, ref_set in data_keyid_sets.items():
        # overlap dataset is reference for overlap exists with base dataset
        print(f"Reference dataset: {ref_name}")
        unq_output[ref_name] = dict()
        for overlap_name, overlap_set in data_keyid_sets.items():
            print(f"\tOverlap dataset: {overlap_name}")
            count_unq_intersect = len(ref_set.intersection(overlap_set))
            perc_unq_interesct = round(count_unq_intersect/len(ref_set), 3)
            print(f"\t% of Reference intersecting Overlap:{perc_unq_interesct}\n")
            unq_output[ref_name][overlap_name] = perc_unq_interesct 

    print(f"Fully unique ouputs: \n{unq_output}\n")
    print_symmetric_table(unq_output, "Intersect\\Reference", "Unique intersection") 

    # reference dataset to be analyzed
    total_output = dict()
    for ref_name, ref_counter in data_keyid_counters.items():
        # overlap dataset is reference for overlap exists with base dataset
        print(f"Reference dataset: {ref_name}")
        total_output[ref_name] = dict()
        for overlap_name, _ in data_keyid_counters.items():
            print(f"\tOverlap dataset: {overlap_name}")
            ref_set, overlap_set = data_keyid_sets[ref_name], data_keyid_sets[overlap_name]
            intersect_ids = ref_set.intersection(overlap_set)
            total_ref_records = len(data_dict[ref_name])
            # count of intersecting records
            count_tot_intersect = sum([
                ref_counter[int_id] for int_id in intersect_ids
            ])
            perc_total_interesct = round(count_tot_intersect/total_ref_records, 3)
            print(f"\tRatio of total intersect to total records: {perc_total_interesct}\n")
            total_output[ref_name][overlap_name] = perc_total_interesct

    print(f"Total output is:\n{total_output}\n")
    print_symmetric_table(total_output, "Intersect\\Reference", "Total intersection")


def update_unq_date_counter(counter:dict, name:str, constraint_id:str, date:str)->dict:
    """This function updates the unq_date_counter by incrementing the count for the constraint 
    in `name` for `date` if the constraint_id is not already in the `date` set.

    Args:
        counter (Dict[
                    name: Dict[
                        date: Dict[
                            "count": int, 
                            "set": Set[constraint_id]
                        ]
                    ]
                ]): 
            dictionary with structure above. For each constraint_name and for each date-bucket (year-month), 
                it has a count of the unique occurances of the `constraint_id` as regulated by the Set of `ids`
        name (str): name of the constraint e.g. "lesson", "line", or "speaker"
        constraint_id (str): id for constraint specified by `name`
        date (str): date string of the year and month in the YYYY-MM format e.g. "2019-08"
    
    Returns:
        (dict): updated counter dict
    """
    # create a date entry if one doesn't exist
    if date not in counter[name]:
        counter[name][date] = dict()
    # create the id-set for the given `date` if it doesn't exist
    if "set" not in counter[name][date]:
        counter[name][date]["set"] = set()
    # if the `constraint_id` is not in the set, increment the date count and add the id to the set
    if constraint_id not in counter[name][date]["set"]:
        counter[name][date]["count"] = counter[name][date].get("count", 0) + 1
        counter[name][date]["set"].add(constraint_id)

    return counter


class DurationAssessor():

    def __init__(self):
        self.datasets = AllDatasets().dataset_list

    def duration_report(self, save_path:str):
        with open(save_path, 'w') as fid:
            for dataset in self.datasets:
                duration = dataset.get_duration()
                name = str(type(dataset))
                out_string = "{0}: {1}\n".format(name, duration)
                fid.write(out_string)
            



class TatoebaAssessor():

    def __init__(self):
        self.dataset = TatoebaDataset()  

    def create_report(self):
        raise NotImplementedError
    
    def audio_by_speaker(self):
        assess_dict = dict()
        audio_files = self.dataset.get_audio_files()

    def test():
        pass
        """
        # steps
        # 1. join eng_sent and audio_sent on 'id' key
        # 2. fitler joined array by `lang`=='eng' to get all English sent with audio
        # 3. do further filtering based on rating and lang ability
        """

        eng_sent_df = pd.read_csv(eng_sent_path, sep='\t', header=None, names=['id', 'lang', 'text'])
        audio_sent_df = pd.read_csv(audio_sent_path, sep='\t', header=None, names=['id', 'user', 'license', 'attr-url']) 

        audio_eng_sent_df = pd.merge(eng_sent_df, audio_sent_df, how='inner', on='id', suffixes=('_eng', '_aud')) 

        user_lang_df = pd.read_csv(user_lang_path, sep='\t', header=None, names=['lang', 'skill', 'user', 'details'])  
        eng_skill_df = user_lang_df[user_lang_df['lang']=='eng']    # shape: (9999, 4)

        audio_eng_skill_df = pd.merge(audio_eng_sent_df, eng_skill_df, how='left', on='user', suffixes=('_m', '_s')) 
        # audio_eng_skill_df.shape = (499085, 9) compared speech_featuresto audio_eng_sent_df.shape = (499027, 6)
        # extra 58 samples I think comes from usernames \N being duplicated 
        # as there are 30 entries in eng_skill_df with username '/N'
        # yeah, audio_eng_skill_df[audio_eng_skill_df['user']=='\\N'].shape   = (60, 9)
        # it is two sentences that are being duplicated across the 30 entries

        audio_eng_skill_df = audio_eng_skill_df.drop_duplicates(subset='id') 
        # audio_eng_skill_df.drop_duplicates(subset='id').shape = (498959, 9)
        # audio_eng_sent_df.drop_duplicates(subset='id').shape = (498959, 6)
        # after drop_duplicates, audio_eng_skill_df[audio_eng_skill_df['user']=='\\N'].shape = (2, 9)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="filters the validated.tsv file based on accent and sentence occurance"
    )
    parser.add_argument(
        "--dataset-name", type=str, help="name of dataset to asses"
    )
    parser.add_argument(
        "--dataset-path", type=str, nargs='*', help="path to json file(s) to parse"
    )
    parser.add_argument(
        "--max-occurance", type=int, default=20, 
        help="max number of times a sentence can occur in output"
    )
    parser.add_argument(
        "--metadata-path", type=str, 
        help="path to metadata.tsv file that contains speaker, line, and lesson ids for speaktrain"
    )
    parser.add_argument(
        "--out-dir", type=str, 
        help="directory where plots and txt files will be saved"
    )
    args = parser.parse_args()

    if args.dataset_name.lower() == "commonvoice":
        assess_commonvoice(args.dataset_path, args.max_occurance)
    elif args.dataset_name.lower() == "speaktrain":
        if args.dataset_path is None:
            use_json = False
        else:
            use_json = True
        assess_speak_train(args.dataset_path, args.metadata_path, args.out_dir, use_json = use_json)
    elif args.dataset_name.lower() == "speakiphone":
        assess_iphone_models(args.dataset_path)
    elif args.dataset_name.lower() == "speak_overlap":
        dataset_overlap(args.dataset_path, args.metadata_path, overlap_key='target_sentence')
    else:
        raise ValueError(f"Dataset name: {args.dataset_name} is not a valid selection")
