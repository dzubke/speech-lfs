"""
this script is meant to assess a dataset along a variety of measures
author: Dustin Zubke
license: MIT
"""
# standard libary
import argparse
from collections import Counter
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
from speech.utils.data_helpers import path_to_id
from speech.utils.io import write_pickle


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



def assess_speak_train(dataset_paths:List[str], tsv_path:str)->None:
    """This function creates counts of the speaker, lesson, and line ids in a speak training dataset
    Args:
        dataset_path (str): path to speak training dataset
        tsv_path (str): path to tsv file that contains speaker, line, and lesson ids 
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


    def _stats(count_dict:dict):
        values = list(count_dict.values())
        mean = round(np.mean(values), 2)
        std = round(np.std(values), 2)
        max_val = round(max(values), 2)
        min_val = round(min(values), 2)
        print(f"mean: {mean}, std: {std}, max: {max_val}, min: {min_val}")
    



    # use this logic for a tsv file
    count_tsv=False
    if count_tsv:
        # count dictionaries for the lesssons, lines, and users (speakers)
        lesson_dict = {} 
        line_dict = {} 
        user_dict ={} 
        # create count_dicts for each
        with open(dataset_path, 'r') as tsv_file: 
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            header = next(tsv_reader) 
            print(header) 
            for row in tsv_reader: 
                lesson_id, line_id, user_id = row[2], row[3], row[4] 
                _increment_key(lesson_dict, lesson_id) 
                _increment_key(line_dict, line_id) 
                _increment_key(user_dict, user_id) 

        # put the labels and count_dicts in list of the for-loop
        constraint_names = ["lesson", "line", "speaker"]
        counter = {
            "lesson": lesson_dict, 
            "line": line_dict, 
            "speaker": user_dict
        }

    # use this logic for a json file supported by a tsv-file
    count_json = True
    if count_json:
        # create mapping from record_id to speaker, line, and lesson ids
        rec_ids_map = dict()
        constraint_names = ['lesson', 'line', 'speaker']
        counter = {name: dict() for name in constraint_names}
        with open(tsv_path, 'r') as tsv_file: 
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            # header: id, text, lessonId, lineId, uid(speaker_id), date
            header = next(tsv_reader)
            for row in tsv_reader:
                rec_ids_map.update({
                    row[0]: {
                        constraint_names[0]: row[2],   # lesson
                        constraint_names[1]: row[3],    # line
                        constraint_names[2]: row[4]     # speaker
                    }
                }

        # iterate through the datasets
        for dataset_path in dataset_paths:
            dataset = read_data_json(dataset_path)
            # iterate through the exmaples in the dataset
            for xmpl in dataset:
                rec_id = path_to_id(xmpl['audio'])
                for name in constraint_names:
                    constraint_id = rec_ids_map[rec_id][name]
                    _increment_key(counter[name], constraint_id)


    # create the plots
    fig, axs = plt.subplots(1,3)
    fig.suptitle('Count')

    # plot and calculate stats of the count_dicts
    for ax, name in zip(axs, constraint_names):
        _plot_count(ax, counter[name], name)
        print(f"{name} stats")
        _stats(counter[name])
        print()
    plt.show()

    #print("unique lessons")
    #print(sorted(list(lesson_dict.keys()))[:200])
    #print(f"number of unique lessons: {len(set(lesson_dict.keys()))}")




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
        r'''
        # skill may not be super helpful in filtering out sentences as nearly all sentences are by skill=5 users
        In [89]: audio_eng_skill_df['skill'].value_counts(sort=True, ascending=False) 
        Out[89]: 
        5     497693
        3         12
        4          7
        \N         4
        Name: skill, dtype: int64

        # not all users are skill 5 in English. It may just be that skill=5 users are the ones recording Eng sentences
        In [90]: eng_skill_df['skill'].value_counts(sort=True, ascending=False) 
        Out[90]: 
        5     4568
        4     2057
        3     1479
        2     1227
        1      430
        \N     195
        0       43

        # like with the Tatoeba subset, CK is 99% of the sentences
        In [91]: audio_eng_skill_df['user'].value_counts(sort=True, ascending=False)   
        Out[91]: 
        CK              494779
        papabear           877
        RB                 805
        Sean_Benward       742
        pencil             348
        jendav             235
        Nero               194
        BE                 178
        dcampbell          167
        mhattick           153
        rhys_mcg           104
        jaxhere             75
        Susan1430           68
        Kritter             58
        Cainntear           38
        MT                  33
        CO                  26
        patgfisher          19
        Source_VOA          18
        samir_t             12
        arh                  7
        Delian               6
        RM                   4
        bretsky              4
        DJT                  4
        LouiseRatty          3
        \N                   2


        # review is not going to helpful because of the total reviews, 99% of them as positive
        In [97]: sent_review_df['review'].value_counts(sort=True, ascending=False) 
        Out[97]: 
        1    1067165
        0       7180
        -1       2949
        '''

    

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
        "--tsv-path", type=str, 
        help="path to tsv file that contains speaker, line, and lesson ids for speaktrain"
    )
    args = parser.parse_args()

    if args.dataset_name.lower() == "commonvoice":
        assess_commonvoice(args.dataset_path, args.max_occurance)
    elif args.dataset_name.lower() == "speaktrain":
        assess_speak_train(args.dataset_path, args.tsv_path)
    elif args.dataset_name.lower() == "speakiphone":
        assess_iphone_models(args.dataset_path)
    else:
        raise ValueError(f"Dataset name: {args.dataset_name} is not a valid selection")
