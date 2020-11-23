# standard libraries
from collections import defaultdict
import glob
import json
import os
import re
import string
import tqdm
# third-party libraries
# project libraries
from speech.utils import convert

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
    transcript = transcript.split()
    for word in transcript:
        phonemes.extend(lexicon.get(word, unk_token))
    
    return phonemes
    

def clean_text(transcript:str)->str:
    """
    This function removes anything that is not alphanumeric, a space, or apostrophe from
    the input `transcript`.
    Args:
        transcript (str): input transcript to be clean
    Returns:
        (str): cleaned transcript
    """
    # allows for alphanumeric characters, space, and apostrophe
    accepted_char = '[^A-Za-z0-9 \']+'
    # replacing weird encodings with apostrophe
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
