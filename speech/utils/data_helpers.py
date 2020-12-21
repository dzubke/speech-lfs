# standard libraries
from collections import defaultdict
import csv
import glob
import json
import os
import re
import string
from typing import Set
# third-party libraries
from prettytable import PrettyTable
import tqdm
# project libraries
from speech.utils import convert
from speech.utils.io import read_data_json

UNK_WORD_TOKEN = list()

def lexicon_to_dict(lexicon_path:str, corpus_name:str=None)->dict:
    """
    This function reads the librispeech-lexicon.txt file which is a mapping of words in the
    librispeech corpus to phoneme labels and represents the file as a dictionary.
    The digit accents are removed from the file name. 
    """
    corpus_names = [
        "librispeech", "tedlium", "cmudict", "commonvoice", "voxforge", "tatoeba", "speaktrain", None
    ]
    if corpus_name not in corpus_names:
        raise ValueError("corpus_name not accepted")
    
    lex_dict = dict()
    with open(lexicon_path, 'r', encoding="ISO-8859-1") as fid:
        lexicon = (l.strip().lower().split() for l in fid)
        for line in lexicon: 
            word, phones = word_phone_split(line, corpus_name)
            phones = clean_phonemes(phones, corpus_name)
            # librispeech: the if-statement will ignore the second pronunciation with the same word
            if lex_dict.get(word, UNK_WORD_TOKEN)  == UNK_WORD_TOKEN:
                lex_dict[word] = phones
    lex_dict = clean_dict(lex_dict, corpus_name)
    #assert type(lex_dict)== defaultdict, "word_phoneme_dict is not defaultdict"
    return lex_dict


def word_phone_split(line:list, corpus_name=str):
    """
    Splits the input list line into the word and phone entry.
    voxforge has a middle column that is not used
    """
    if corpus_name == "voxforge":
        word, phones = line[0], line[2:]
    else:
        word, phones = line[0], line[1:]
    return word, phones


def clean_phonemes(phonemes, corpus_name):

    if corpus_name == "librispeech" or corpus_name == "cmudict":
        return list(map(lambda x: x.rstrip(string.digits), phonemes))
    else:
        return phonemes


def clean_dict(lex_dict, corpus_name):
    corpus_num_duplicates = ["tedlium", "cmudict", "voxforge"]
    if corpus_name in corpus_num_duplicates:
        return defaultdict(lambda: UNK_WORD_TOKEN, 
                {key: value for key, value in lex_dict.items() if not re.search("\(\d\)$", key)})
    else: 
        return lex_dict


def combine_lexicons(lex1_dict:dict, lex2_dict:dict)->(dict, dict):
    """
    this function takes as input a dictionary representation of the two
    lexicons and outputs a combined dictionary lexicon. it also outputs
    a dict of words with different pronunciations
    Arguments:
        lex1_dict - dict[str:list(str)]: dict representation of the first lexicon
        lex2_dict - dict[str:list(str)]: dict representation of the second lexicon
    Returns:
        combo_dict - dict[str:list(str]
    """

    word_set = set(list(lex1_dict.keys()) + list(lex2_dict.keys()))
    combo_dict = defaultdict(lambda: list())
    diff_labels = dict()

    for word in word_set:
        if  word not in lex1_dict:
            # word has to be in lex2_dict
            combo_dict.update({word:lex2_dict.get(word)})
        elif word not in lex2_dict:
            # word has to be in lex1_dict
            combo_dict.update({word:lex1_dict.get(word)})
        else:
            # word is in both dicts, used lex2_dict
            if lex1_dict.get(word) == lex2_dict.get(word):
                combo_dict.update({word:lex2_dict.get(word)})
            else:   # phoneme labels are not the same
                combo_dict.update({word:lex2_dict.get(word)})
                diff_labels.update({word: {"lex1": lex1_dict.get(word), "lex2": lex2_dict.get(word)}})
    # print(f"words with different phoneme labels are: \n {diff_labels}")
    print(f"number of words with different labels: {len(diff_labels)}")

    return  combo_dict, diff_labels


def create_lexicon(cmu_dict:dict, ted_dict:dict, lib_dict:dict, out_path:str='')->dict:
    """
    Creates a master lexicon using pronuciations from first cmudict, then tedlium
    dictionary and finally librispeech. 
    Arguments:
        cmu_dict - dict[str:list(str)]: cmu dict processed with lexicon_to_dict
        ted_dict - dict[str:list(str)]: tedlium dict processed with lexicon_to_dict
        lib_dict - dict[str:list(str)]: librispeech dict processed with lexicon_to_dict
        out_path - str (optional): output path where the master lexicon will be written to
    Returns:
        master_dict - dict[str:list(str)]
    """

    word_set = set(list(cmu_dict.keys()) + list(ted_dict.keys())+list(lib_dict.keys()))
    master_dict = defaultdict(lambda: UNK_WORD_TOKEN)

    # uses the cmu_dict pronunciation first, then tedlium_dict, and last librispeech_dict
    for word in word_set:
        if  word in cmu_dict:
            master_dict.update({word:cmu_dict.get(word)})
        elif word in ted_dict:
            master_dict.update({word:ted_dict.get(word)})
        elif word in lib_dict:
            master_dict.update({word:lib_dict.get(word)})

    if out_path != '': 
        sorted_keys = sorted(master_dict.keys())
        with open(out_path, 'w') as fid:
            for key in sorted_keys:
                fid.write(f"{key} {' '.join(master_dict.get(key))}\n")
 
    return master_dict


def skip_file(dataset_name:str, audio_path:str)->bool:
    """
    if the audio path is in one of the noted files with errors, return True
    """

    sets_with_errors = ["tatoeba", "voxforge", "speaktrain"]
    # CK is directory name and min, max are the ranges of filenames
    tatoeba_errors = {"CK": {"min":6122903, "max": 6123834}}
    voxforge_errors = {"DermotColeman-20111125-uom": "b0396"}

    skip = False
    if dataset_name not in sets_with_errors:
        # jumping out of function to reduce operations
        return skip
    file_name, ext = os.path.splitext(os.path.basename(audio_path))
    dir_name = os.path.basename(os.path.dirname(audio_path))
    if dataset_name == "tatoeba":
        for tat_dir_name in tatoeba_errors.keys():
            if dir_name == tat_dir_name:
                if tatoeba_errors[tat_dir_name]["min"] <= int(file_name) <=tatoeba_errors[tat_dir_name]["max"]:
                    skip = True
   
    elif dataset_name == "voxforge":
        #example path: ~/data/voxforge/archive/DermotColeman-20111125-uom/wav/b0396.wv
        speaker_dir = os.path.basename(os.path.dirname(os.path.dirname(audio_path)))
        if speaker_dir in voxforge_errors.keys():
            file_name, ext = os.path.splitext(os.path.basename(audio_path))
            if file_name in voxforge_errors.values():
                skip = True
    
    elif dataset_name == "speaktrain":
        # the speak files in the test sets cannot go into the training set
        # so they will be skipped based on their firestore record id
        speak_test_ids = set(get_speak_test_ids())
        if file_name in speak_test_ids:
            skip = True

    return skip


def get_files(root_dir:str, pattern:str):
    """
    returns a list of the files in the root_dir that match the pattern
    """
    return glob.glob(os.path.join(root_dir, pattern))


def get_speak_test_ids():
    """
    returns the document ids of the recordings in the old (2019-11-29) and new (2020-05-27) speak test set.
    Two text files containing the ids must existing in the <main>/speech/utils/ directory.
    """
    abs_dir = os.path.dirname(os.path.abspath(__file__))

    file_path_2019 = os.path.join(abs_dir, 'speak-test-ids_2019-11-29.txt')
    file_path_2020 = os.path.join(abs_dir, 'speak-test-ids_2020-05-27.txt')

    assert os.path.exists(file_path_2020), \
        "speak-test-ids_2020-05-27.txt doesn't exist in <main>/speech/utils/"
    assert os.path.exists(file_path_2019), \
        "speak-test-ids_2019-11-29.txt doesn't exist in <main>/speech/utils/"

    with open(file_path_2019, 'r') as id_file:
        ids_2019 = id_file.readlines() 
        ids_2019 = [i.strip() for i in ids_2019]

    with open(file_path_2020, 'r') as id_file: 
        ids_2020 = id_file.readlines() 
        ids_2020 = [i.strip() for i in ids_2020] 

    return ids_2019 + ids_2020


def text_to_phonemes(transcript:str, lexicon:dict, unk_token=list())->list:
    """
    The function takes in a string of text, cleans the text, and outputs a list of phoneme 
    labels from the `lexicon_dict`. 
    Args:
        transcript (str): string of words
        lexicon (dict): lexicon mapping of words (keys) to a list of phonemes (values)
    Returns:
        (list): a list of phoneme strings
    """
    if isinstance(unk_token, str):
        unk_token = [unk_token]
    elif isinstance(unk_token, list):
        pass
    else:
        raise TypeError(f"unk_token has type {type(unk_token)}, not str or list")

    phonemes = list()
    transcript = clean_text(transcript)
    transcript = transcript.split(' ')
    for word in transcript:
        phonemes.extend(lexicon.get(word, unk_token))
    
    return phonemes


def process_text(transcript:str)->str:
    """This function removes punctuation (except apostrophe's) and extra space
    from the input `transcript` string and lowers the case. 

    Args:
        transcript (str): input string to be processed
    Returns:
        (str): processed string
    """
    # allows for alphanumeric characters, space, and apostrophe
    accepted_char = '[^A-Za-z0-9 \']+'
    # replacing apostrophe's with weird encodings
    transcript = transcript.replace(chr(8217), "'")
    # filters out unaccepted characters, lowers the case
    try:
        transcript = transcript.strip().lower()
        transcript = re.sub(accepted_char, '', transcript)
    except TypeError:
        print(f"Type Error with: {transcript}")
    # check that all punctuation (minus apostrophe) has been removed 
    punct_noapost = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    for punc in punct_noapost:
        if punc in transcript:
            raise ValueError(f"unwanted punctuation {punc} in transcript")
  
    return transcript


def check_update_contraints(record_id:int, 
                            record_ids_map:dict,
                            id_counter:dict, 
                            constraints:dict)->bool:
    """This function is used by downloading and filtering code primarily on speak data
    to constrain the number of recordings per speaker, line, and/or lesson.
    It checks if the counts for the `record_id` is less than the constraints in `constraints. 
    If the count is less, the constraint passes and the `id_counter` is incremented.
  
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


def check_disjoint_filter(record_id:str, disjoint_id_sets:dict, record_ids_map:dict)->bool:
    """This function checks if the record_id contains any common ids with the disjoint datasets.
    If a common ids is found, the check fails.

    This function is used in filter.py.

    Args:
        record_ids (str): record id for a recording.
        disjoint_id_sets (Dict[str, Set[str]]): dictionary that maps the ids along which the output dataset
            will be disjoint to the set of ids included in the `disjoint_datasets`. 
        record_ids_map (Dict[str, Dict[str, str]]): dictionary to maps record_id to other ids like
            speaker, lesson, line (or target-sentence).
        
    Returns:
        (bool): True if the ids associated with the record_id are not contained in any of 
            the `disjoint_ids_sets`. Otherwise, False.
    """
    # assumes the check passes (not the safest initial assumption but it makes the logic cleaner)
    pass_check = True
    # names of the ids along which the output dataset will be disjoint
    for id_name, dj_id_set in disjoint_id_sets.items():
        disjoint_id = record_ids_map[record_id][id_name]
        # if the id is contained in the id_set of the disjoint_datasets, the check fails
        if disjoint_id in dj_id_set:
            pass_check = False
            break
    
    return pass_check


def get_dataset_ids(dataset_path:str)->Set[str]:
    """This function reads a dataset path and returns a set of the record ID's
    in that dataset. The record ID's mainly correspond to recordings from the speak dataset. 
    For other datsets, this function will return the filename without the extension.

    Args:
        dataset_path (str): path to the dataset
    
    Returns:
        Set[str]: a set of the record ID's
    """
    # dataset is a list of dictionaries with the audio path as the value of the 'audio' key.
    dataset = read_data_json(dataset_path)

    return set([path_to_id(xmpl['audio']) for xmpl in dataset])


def path_to_id(record_path:str)->str:
        #returns the basename of the path without the extension
        return os.path.basename(
            os.path.splitext(record_path)[0]
        )


def get_record_id_map(metadata_path:str, id_names:list=None)->dict:
    """This function returns a mapping from record_id to other ids like speaker, lesson,
    line, and target sentence. This function runs on recordings from the speak firestore database.

    Args:
        metadata_path (str): path to the tsv file that contains the various ids
        id_names (List[str]): names of the ids in the output dict. 
            This is currented hard-coded to the list: ['lesson', 'target-sentence', 'speaker']
    
    Returns:
        Dict[str, Dict[str, str]]: a mapping from record_id to a dict
            where the value-dict's keys are the id_name and the values are the ids
    """
    assert os.path.splitext(metadata_path)[1] == '.tsv', \
        f"metadata file: {metadata_path} is not a tsv file"

    # check that input matches the expected values
    # TODO: hard-coding the id-names isn't flexible but is the best option for now
    expected_id_names = ['lesson', 'target_sentence', 'speaker']
    if id_names is None:
        id_names = expected_id_names
    assert id_names == expected_id_names, \
        f"input id_names: {id_names} do not match expected values: {expected_id_names}"

    # create a mapping from record_id to lesson, line, and speaker ids
    expected_row_len = 7
    with open(metadata_path, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        header = next(tsv_reader)
        # this assert helps to ensure the row indexing below is correct
        assert len(header) == expected_row_len, \
            f"Expected metadata header length: {expected_row_len}, got: {len(header)}."
        # header: id, text, lessonId, lineId, uid(speaker_id), redWords_score, date
        print("header: ", header)

        # mapping from record_id to other ids like lesson, speaker, and line
        record_ids_map = dict()
        for row in tsv_reader:
            assert len(row) == expected_row_len, \
                f"row: {row} is len: {len(row)}. Expected len: {expected_row_len}"
            tar_sentence = process_text(row[1])
            record_ids_map[row[0]] = {
                    "record": row[0],           # adding record for disjoint_check
                    id_names[0]: row[2],        # lesson
                    id_names[1]: tar_sentence,  # using target_sentence instead of lineId
                    id_names[2]: row[4]         # speaker
            }

    return record_ids_map



def print_symmetric_table(values_dict:dict, row_name:str, title:str)->None:
    """Prints a table of values in  2-d dict with identical inner and outer keys
    Args:
        values_dict (Dict[str, Dict[str, float]]): 2-d dictionary with identical keys on the two levels
        row_name (str): name of the rows
        title (str): title of the table
    """
    table = PrettyTable(title=title)
    sorted_keys = sorted(values_dict.keys())
    table.add_column(row_name, sorted_keys)
    for data_name in sorted_keys:
        table.add_column(data_name, [values_dict[data_name][key] for key in sorted_keys])
    print(table)



def print_nonsym_table(values_dict:dict, row_name:str, title:str)->None:                
    """Prints a prety table from a 2-d dict that has different inner and outer keys (not-symmetric)
    Args: 
        values_dict (Dict[str, Dict[str, float]]): 2-d dict with different keys on inner and outer levels 
        row_name (str): name of the rows 
        title (str): title of the table 
    """ 
    single_row_name = list(values_dict.keys())[0] 
    sorted_inner_keys = sorted(values_dict[single_row_name].keys()) 
    column_names = [row_name] + sorted_inner_keys 
    table = PrettyTable(title=title, field_names=column_names)                             
                                    
    for row_name in values_dict: 
        table.add_row([row_name] + [values_dict[row_name][key] for key in sorted_inner_keys]) 
    print(table)