# this file is a hodge-podge of scripts used to process lexicons and datasets using the 
# montreal forced aligner (MFA). the MFA is used provide potentially better phoneme labels
# Copyright: Speak Labs 2021
# Author: Dustin Zubke

import argparse
from collections import Counter, defaultdict
import copy
import csv
import fnmatch
import glob
import json
import os
from pathlib import Path
import re
import shutil
import string
from typing import Dict, List, Tuple
# third party libraries
import editdistance as ed
import numpy as np
import tqdm
# project libraries
from speech.utils import textgrid
from speech.utils.data_helpers import clean_phonemes, get_record_ids_map, path_to_id, process_text
from speech.utils.io import read_data_json, write_data_json



def compute_lexicon_outliers(lexicon_path:str):
    """This function computes the outliers in the lexicon by length of pronunciation. It is meant
    to catch very short or long pronunciations.
    
    It does this by printing pronunciations that are more than 2 standard deviations away from the 
    mean ratio of the word length to pronunciation length, where word length is the number of 
    characters and pronunciation length is the number of phonemes. 
    
    Args:
        lexicon_path (str): path to lexicon
    """

    # returns lexicon with pronunciations as list of phoneme strings
    lex_dict = load_lex_dict(lexicon_path, split_phones=True)

    pronun_ratios  = [len(word)/len(phones) for word, phones in lex_dict.items()]
    mean_ratio = np.mean(pronun_ratios)
    stddev_ratio = np.std(pronun_ratios)

    outlier_factor = 4.0    # num std-deviations that define an outlier
    lower_bound = mean_ratio - outlier_factor * stddev_ratio
    upper_bound = mean_ratio + outlier_factor * stddev_ratio

    print(f"mean: {mean_ratio}, std-dev: {stddev_ratio}, # of stddev for outlier: {outlier_factor}")

    outliers = defaultdict(list)
    for word, phones in lex_dict.items():
        ratio = len(word)/ len(phones)
        if ratio < lower_bound or ratio > upper_bound:
            outliers[word].extend(phones)

    print(f"number of outliers outside bounds: {len(outliers)}")
    for word, phone_list in outliers.items():
        for phones in phone_list:
            print(f"{word} {phones}")


def create_spk_dir_tree(spk_training_jsons:List[str], audio_dir:str)->None:
    """It seems the mfa aligner tool will only parallelize directory trees, so the audio being
    all in a single dir does not run in parallel. Given this, this function converts that audio
    paths from being in a single dir, `audio_dir`, to a single-level tree where each subdir of
    `audio_dir` will have a most 1000 audio files. 
    
    The list of training jsons `spk_training_jsons` will be re-written to use the updated paths 
    to the new subdirectories. 

    Note: both .wav and .txt files will be moved. The .txt files are used by the mfa aligner

    Args:
        spk_training_jsons (List[str]): List of paths to relevent speak training jsons
        audio_dir (str): path to directory that contains all speak training files. 
            These files will be moved into subdirectories under `audio_dir`
    """
    # number of wav files in each subdirectory
    SUB_DIR_SIZE = 1000
    
    # list and sort .wav files in audio_dir
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    
    # create a dict for each training_json whose values is a dict mapping audio paths to examples
    json_dict = dict()
    for json_path in spk_training_jsons:
        json_dict[json_path] = {
            xmpl['audio']: xmpl for xmpl in read_data_json(json_path)
        }
    
    file_count = 0
    subdir_count = 0
    for audio_file in audio_files:
        # if file_count has filled up the previous sub-dir, create a new one
        if file_count % SUB_DIR_SIZE == 0:
            subdir_count += 1
            os.makedirs(os.path.join(audio_dir, str(subdir_count)), exist_ok=True)

        # inserts the subdir_count into the audio_file path
        audio_dir, basename = os.path.split(audio_file)
        new_audio_file = os.path.join(audio_dir, str(subdir_count), basename)
        
        # update the path names in each training json
        for xmpl_dict in json_dict.values():
            if audio_file in xmpl_dict:
                # update the audio path
                xmpl_dict[audio_file]['audio'] = new_audio_file
        
        # move the .wav and .txt files to the new sub-dir
        shutil.move(audio_file, new_audio_file)
        if os.path.exists(audio_file.replace(".wav", ".txt")):
            shutil.move(audio_file.replace(".wav", ".txt"), new_audio_file.replace(".wav", ".txt"))

        file_count += 1

    for json_path, xmpl_dict in json_dict.items():
        # TODO (drz): remove renaming of `json_path` once verifying the script works as expected
        json_path += "-new"
        write_data_json(xmpl_dict.values(), json_path)
 

def create_spk_transcripts(spk_metadata_path:str, spk_audio_dir:str, lex_path:str):
    """This function creates an uppercase transcript as an individual .txt file for each audio file
    in `audio_dir` using the `spk_metadata_path`. It does not create a transcript for files with words
    not included in the lexicon from `lex_path`. 
    """
    record_ids_map = get_record_ids_map(spk_metadata_path, has_url=True)
    lex_dict = load_lex_dict(lex_path)
    lex_words = set([word for word in lex_dict])
    
    examples = {
        "total": 0,
        "oov": 0
    }
    audio_files = glob.glob(os.path.join(spk_audio_dir, "*.wav"))
    for audio_file in tqdm.tqdm(audio_files):
        file_id = path_to_id(audio_file)
        # trancript is already processed in `get_record_ids_map`
        transcript = record_ids_map[file_id]['target_sentence']
        transcript = transcript.upper().split()
        # checks if the transcript has an out-of-vocab word
        has_oov = any([(word not in lex_words) for word in transcript])
        if has_oov:
            examples['oov'] += 1
            continue
        examples['total'] += 1
        # write the transcript to a txt file
       # txt_file = audio_file.replace(".wav", ".txt")
       # with open(txt_file, 'w') as fid:
       #     fid.write(" ".join(transcript))
    print(f"num oov_examples: {examples['oov']} out to total: {examples['total']}")    



def insert_mispronunciations(lex_path:str, spk_word_path:str, save_path:str):
    """This function adds additional pronunciations created by replacing certain phonemes for words
    in the speak dataset. The existing and additional pronunciations are saved to `save_path`.
    """
    # phoneme keys in the dict will be replaced with the phonmes in the values list
    phoneme_swaps = {
        'AE1':['EH1'],
        'B': ['P'],
        'D': ['D IY1', 'D AH1', 'T'],
        'DH' : ['D'], 
        'F': ['P'],
        'G': ['K']  ,
        'IH1': ['IY1'],
        'IY1': ['IH1'],
        'L' : ['R'],
        'P': ['F'],
        'R' : ['L'],
        'R AO1': ['AO1', 'R'],
        'S': ['SH'],
        'SH': ['S'],
        'TH': ['S'],
        'UH1': ['UW1'],
        'V': ['B'],
        'W UH1': ['UH1'],
        'W UW1': ['UW1'],
        'Z': ['JH', 'CH'],
    }

    lex_dict = load_lex_dict(lex_path, split_phones=True)
    spk_upper_words = load_spk_upper_words(spk_word_path)
    for word in spk_upper_words:
        # make all additions to a word in a separate list to prevent an infinite-loop
        new_pronunciations = list()
        # for each pronnciation of `word`
        for lex_phones in lex_dict[word]:
            # for each swap-phoneme as `src_phone`
            for src_phone, dst_phone_list in phoneme_swaps.items():
                if src_phone in lex_phones:
                    # for each possible `dst_phone`, swap `dst_phone` for `src_phone`
                    for dst_phone in dst_phone_list:
                        new_phones = copy.copy(lex_phones)
                        new_phones = [dst_phone if x==src_phone else x for x in lex_phones]
                        new_pronunciations.append(new_phones) 
                          
        lex_dict[word].extend(new_pronunciations) 

    save_lex_dict(lex_dict, save_path, split_phones=True)


def phoneme_occurance(lex_path:str, save_path:str)->None:
    """This function computes an occurance count of the phonemes in the lexicon
    """

    lex_dict = load_lex_dict(lex_path)

    phone_counter = Counter()
    for word, phone_str_list in lex_dict.items():
        for phone_str in phone_str_list:
            phones = phone_str.strip().split(' ')
            phone_counter.update(phones)

    with open(save_path, 'w') as fid:
        for phone, count in phone_counter.most_common():
            fid.write(f"{phone} {count}\n")


def manual_entry(lex_path:str, save_path:str):
    """This function will manually add a few mispronunciations to the input lexicon
    """

    manual_entries = {
        "TRAVELING": "T R AE1 V ER0  L IH0 NG",        
    }



def remove_suffix(lex_path:str, spk_words_path:str, save_path:str):
    """A pronunciation mistake is to drop the 'ed' off the end of a word. 
    This function will add the pronunciation of an 'ed'-less word as a word's
    pronunciation. For example the pronunciation of 'dream' will be used for the
    word 'dreamed'.
    This addition will only be made for words in the `spk_words_path` file. 
    """

    lex_dict = load_lex_dict(lex_path)

    # character lengths of the phonemes for each suffix
    # allows for indexing as [-len:] to remove suffix phonemes
    suffix_lengths = {
        "ED": 2,
        "ING": 7
    }
    
    spk_upper_words = load_spk_upper_words(spk_words_path)

    for word in spk_upper_words:
        if word == 'GOING': # `G OW1 IH0 N` pronunciation is edge-case
            continue

        for suffix, suffix_len in suffix_lengths.items():
            if word.endswith(suffix):
                new_pronunciations = list()
                for phones in lex_dict[word]:
                    new_phones = copy.copy(phones)
                    new_pronunciations.append(new_phones[:-suffix_len])
                
                lex_dict[word].extend(new_pronunciations)
                        

    save_lex_dict(lex_dict, save_path)


def load_spk_upper_words(spk_words_path:str)->List[str]:
    """This funciton returns a list of the uppecase words in the speak training set
    """
    spk_words = list()    
    with open(spk_words_path, 'r') as fid:
        for row in fid:
            word, _ = row.strip().split()
            spk_words.append(word.upper())

    return spk_words
                

def load_lex_dict(lex_path:str, split_phones=False)->Dict[str, List[str]]:
    """This function reads the lexicon path and returns a dictionary with 
    the uppercase words as keys and values as a list of one or more pronunciations as strings.
    
    Args:
        lex_path (str): path to lexicon
        split_phones (bool): if true, the phonemes will be split into a list of strings, default False    
    """

    
    lex_dict = defaultdict(list)
    rows = []
   
    # ensure all entries are space-separated, rather than tab-separated 
    with open(lex_path, 'r') as fid:
        for row in fid:
            rows.append(row.replace('\t', ' '))

    for row in rows:
        row = row.strip().split(' ', maxsplit=1)
        # check of unexpected rows
        if len(row) != 2:
            print(f"short row: {row}")
            continue
        word, phones = row
        phones = phones.strip()
        if split_phones:
            phones = phones.split(' ')
        lex_dict[word].append(phones)

    return lex_dict    


def save_lex_dict(lex_dict:dict, save_path:str, split_phones=False):
    """Save the lex dict to the save_path

    Args:
        lex_dict (dict): dictionary of pronunications either as strings or list of strings
        save_path (str): output path for lexicon
        split_phnoes (bool): if True, values in lex_dict are a list of list of phoneme-strings,
            if False, values are list of single-strings of all phonemes
    """
    with open(save_path, 'w') as fid:
        for key in lex_dict:
            for phones in lex_dict[key]:
                if split_phones:
                    phones = " ".join(phones)
                fid.write(f"{key} {phones}\n")


def load_aligner_phones_lower(aligner_phone_path:str)->Dict[str,List[str]]:
    """Returns a dict mapping example_id to lowercase phonemes based on the input path to the
    aligner's phonemes.

    Args:
        aligner_phone_path (str): path to file output by `extract_phonemes` function
    
    Returns:
        (dict)
    """   
    phone_path = Path(aligner_phone_path)

    aligner_phones = dict()
    for row in phone_path.read_text().split('\n'):
        row = row.strip().split()
        if len(row) < 2:
            print(f"row: {row} as no phonemes")
            continue 
        
        file_id, phones = row[0], row[1:]
        # remove the digit and lower case
        phones = [phone.rstrip(string.digits).lower() for phone in phones]
        aligner_phones[file_id] = phones
    
    return aligner_phones


def expand_contractions(lex_path:str, contractions_path:str, save_path:str):
    """This function adds entries to the output lexicon that expands contractions.
    For example the pronunciation of "i'll" will now have a new entry for the phonemes of
    the phrase "i will". The pronunciations of the expanded contraction will be added to the output
    lexicon.
    
    Args:
        lex_path (str): path to lexicon
        contractions_path (str): path to file with contraction-to-expansion mapping
        save_path (str): path to output lexicon
    """
    # create a lexicon dict where pronunciations are strings in a list
    # words with multiple pronunciations have len(list) > 1
    lex_dict = load_lex_dict(lex_path)
    
    # for each contraction-expansion pair, create a new entry in the lex_dict
    with open(contractions_path, 'r') as fid:
         for row in fid:
            row = row.strip().upper().split(' ')
            contraction, expansion = row[0], row[1:]
            
            # the combinations below only work for 2-word expansions
            assert len(expansion) == 2, f"expansion: {expansion} is not size 2"
            new_pronun = list()
            for phones_1 in lex_dict[expansion[0]]:
                for phones_2 in lex_dict[expansion[1]]:
                    lex_dict[contraction].append(phones_1 + " " + phones_2)

    save_lex_dict(lex_dict, save_path)      


def combine_cmud_libsp_lexicons(cmu_path:str, libsp_path:str, save_path:str)->None:
    """This function combines the cmudict and librispeech lexicons and writes the combined
        file to the file in `save_path`. For cmudict, it removes the digits "(1)" demarking
        multiple pronunciations.
    
    Args:
        cmu_path (str): path to cmudict
        libsp_path (str): path to librispeech
        save_path (str): path to output file
    """

    # reads inputs and remove digit marker from cmu
    word_phone_list = list()
    with open(cmu_path, 'r') as cmuid:
        for row in cmuid:
            word_phones = row.strip().split(' ', maxsplit=1)
            if len(word_phones) != 2:
                print(f"cmu: unexpected row size in row:  {word_phones}")
                continue
            word, phones = word_phones
            # remove the "(1)" marker in alternate pronunciations
            word = re.sub("\(\d\)", '', word)
            word_phone_list.append((word, phones))

    with open(libsp_path, 'r') as libid:
        for row in libid:
            word_phones = row.strip().split(' ', maxsplit=1)
            if len(word_phones) != 2:
                # some entries are tab-delimited
                word_phones = row.strip().split('\t', maxsplit=1)
                if len(word_phones) != 2:
                    print(f"libsp: unexpected row size in row:  {word_phones}")
                    continue
            word, phones = word_phones   
            word_phone_list.append((word, phones))

    # pass list through set to  de-duplicate
    word_phone_list = sorted(set(word_phone_list))
    
    # write to file
    with open(save_path, 'w') as fid:
        for word, phones in word_phone_list:
            fid.write(f"{word} {phones}\n")


def spk_word_count(metadata_file:str, out_path:str):
    """This funciton creates a count of all the words in the speak training set.
    Args:
        metadata_file (str): path to the metadata file with the word targets
        out_path (str): file where the count will be saved
    """

    word_counter = Counter()
    with open(metadata_file, 'r') as fid:
        reader = csv.reader(fid, delimiter='\t')
        header = next(reader)
        for row in tqdm.tqdm(reader, total=3.165e7):
            target = process_text(row[1])
            word_counter.update(target.split(' '))
            
    with open(out_path, 'w') as fid:
        for word, count in word_counter.most_common():
            fid.write(f"{word} {count}\n")


def update_train_json(old_json_path:str, aligner_phones_path:str, new_json_path:str):
    """Saves a new training json that replaces the phones in the existing training json at 
    `old_json_path` with the aligner phonemes in `aligner_phones_path`. The new training json
    is saved to `new_json_path`. 

    Args:
        old_json_path (str): path to existing training json
        aligner_phones_path (str): path to aligner phonemes
        new_json_path (str): path where new training json will be saved
    
    """

    aligner_phones = load_aligner_phones_lower(aligner_phones_path)

    # train_json is list of dicts with keys: 'audio', 'duration', 'text'
    train_json = read_data_json(old_json_path)
    
    # update train_json with new phonemes in place
    for idx, xmpl in enumerate(train_json):
        audio_id = path_to_id(xmpl['audio'])
        if audio_id in aligner_phones:
            xmpl['text'] = aligner_phones[audio_id]
        # remove examples not in the aligner_phones. edit this design choice, if desired. 
        else:
          train_json.pop(idx)

    write_data_json(train_json, new_json_path) 


def compare_phonemes(aligner_phone_path:str, training_json_path:str, save_path:str)->None:
    """This function calculates the levenshtein distance between the new aligner phonemes
    and the existing phoneme labels and writes the sorted file-id and distances to `save_path`

    Args:
        aligner_phone_path (str): path to file with space separate file_id and aligner phonemes pairs
        training_json_path (str): patht to standard training json file
        save_path (str): path where sorted distances will be written
    """

    
    # load the lower-case aligner phones
    aligner_phones = load_aligner_phones_lower(aligner_phone_path)
    
    data = read_data_json(training_json_path)
    
    # records the distance between old and new phonemes
    distance_dict = dict()
    for xmpl in data:
        old_phones = xmpl['text']
        file_id = path_to_id(xmpl['audio'])
        
        try:
            new_phones = aligner_phones[file_id]
        except KeyError:
            print(f"file_id {file_id} not found in aligner phones. skipping")
            continue

        distance = ed.eval(old_phones, new_phones)
        distance_dict[xmpl['audio']] = (distance, old_phones, new_phones)

    # sort examples by distance 
    sorted_dist = sorted(distance_dict.items(), key=lambda x: x[1][0], reverse=True)
    
    # summary stats
    total_dist = 0
    total_old_phones = 0

    # write formatted output to `save_path`
    with open(save_path, 'w') as fid:
        for file_path, (dist, old_phones, new_phones) in sorted_dist:
            fid.write(f"file:       {file_path}\n")
            fid.write(f"dist:       {dist}\n") 
            fid.write(f"old_phones: {old_phones}\n")
            fid.write(f"new_phones: {new_phones}\n")
            fid.write("\n")

            total_dist += dist
            total_old_phones += len(old_phones)

    # print summary stats
    print(f"total distance: {total_dist} out of {total_old_phones} total phonemes")
    print(f"average distance: {round(total_dist/total_old_phones, 3)}")


def extract_aligner_phonemes(data_dir:str, save_path:str):
    """This function gathers all textgrid files whose parent dir is `data_dir`, extracts
        the phonemes from each textgrid file, and writes only the phonemes to `save_path`
        as a space-separate format prefixed by the filename.
    """
    
    file_phones = dict()
    remove_markers = ["sil", "sp", "spn"]
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in fnmatch.filter(filenames, "*.TextGrid"):
            tg_file = os.path.join(dirpath, filename)
            # create a TextGrid object
            tg_object = textgrid.TextGrid.load(tg_file)
            for tier in tg_object:
                if tier.nameid == "phones":
                    start_end_phone = tier.make_simple_transcript()
                    phones = [elem[2] for elem in start_end_phone]
                    # remove the 'sil', 'sp', and 'spn', markers
                    phones = [phone for phone in phones if phone not in remove_markers] 
                    file_phones[filename] = " ".join(phones)
    
    
    with open(save_path, 'w') as fid:
        for filename, phones in file_phones.items():
            filename = filename.replace(".TextGrid", "")
            fid.write(f"{filename} {phones}\n")    
    


def prep_librispeech():
    libsp_glob = "/mnt/disks/data_disk/data/LibriSpeech/**/**/**/*trans.txt" 
    
    for trans_file in glob.glob(libsp_glob):
        dir_name = os.path.dirname(trans_file)
        with open(trans_file, 'r') as rfid:
            for row in rfid:
                filename, transcript = row.strip().split(' ', maxsplit=1)
                out_file = os.path.join(dir_name, filename)
                os.remove(out_file)
                out_file = out_file + ".txt"
                with open(out_file, 'w') as wfid:
                    wfid.write(transcript)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--action", help="determines what function to call"
    )
    parser.add_argument(
        "--data-paths", nargs="+",
        help="Paths to relevant function data. A single path can be used."
    )
    parser.add_argument(
        "--audio-dir", help="Paths to directory that contains audio."
    )
    parser.add_argument(
        "--save-path", help="path to output file"
    )

    args = parser.parse_args()

    if args.action == "extract-phonemes":
        extract_aligner_phonemes(*args.data_paths, args.save_path) 
    elif args.action == "spk-word-count":
        spk_word_count(*args.data_paths, args.save_path)
    elif args.action == "combine-cmu-libsp":
        combine_cmud_libsp_lexicons(*args.data_paths, save_path=args.save_path)
    elif args.action == "expand-contractions":
        expand_contractions(*args.data_paths, save_path=args.save_path)
    elif args.action ==  "remove-suffix":
        remove_suffix(*args.data_paths, save_path=args.save_path)
    elif args.action ==  "phoneme-occurance":
        phoneme_occurance(*args.data_paths, save_path=args.save_path)
    elif args.action == "insert-mispronunciations":
        insert_mispronunciations(*args.data_paths, save_path=args.save_path)
    elif args.action == "create-spk-transcripts":
        create_spk_transcripts(*args.data_paths)
    elif args.action == "compare-phonemes":
        compare_phonemes(*args.data_paths, save_path=args.save_path)
    elif args.action == "create-subdirs":
        create_spk_dir_tree(args.data_paths, args.audio_dir)
    elif args.action == "compute-lex-outliers":
        compute_lexicon_outliers(*args.data_paths)
    elif args.action == "update-train-json":
        update_train_json(*args.data_paths, args.save_path)
    else:
        raise ValueError(f"action: {args.action} not an accepted function action")
