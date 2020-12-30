# standard libary
import argparse
from collections import defaultdict, Counter
import csv
from datetime import date
from functools import partial
import glob
import io
import json
import logging
import math
import multiprocessing as mp
import os
import re
import subprocess
from tempfile import NamedTemporaryFile
import time
from typing import Tuple
import unicodedata
import urllib
# third party libraries
import tqdm
import yaml
# project libraries
from speech.utils import data_helpers, wave, convert
from speech.utils.data_helpers import process_text
from speech.utils.data_helpers import (
    check_disjoint_filter, check_update_contraints, get_disjoint_sets, get_record_ids_map, 
    lexicon_to_dict, skip_file
)

logging.basicConfig(filename=None, level=10)

class DataPreprocessor(object):
    
    def __init__(self, 
                 dataset_dir:str, 
                 dataset_files:dict,
                 dataset_name:str, 
                 lexicon_path:str,
                 force_convert:bool, 
                 min_duration:float, 
                 max_duration:float,
                 download_audio:bool):

        self.dataset_dir = dataset_dir
        self.dataset_dict = dataset_files
        if lexicon_path !='':
            self.lex_dict = lexicon_to_dict(lexicon_path, dataset_name.lower())
        else: 
            self.lex_dict = None
        # list of tuples of audio_path and transcripts
        self.audio_trans=list()
        # if true, all audio wav files will be overwritten if they exists
        self.force_convert = force_convert
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_ext = 'wav'
        self.download_audio = download_audio


    def process_datasets(self):
        """
        This function iterates through the datasets in dataset_dict and calls the self.collect_audio_transcripts()
        which stores the audio_path and string transcripts in the self.audio_trans object. 
        Then, this function calls self.write_json() writes the audio and transcripts to a file.
        """
        raise NotImplementedError

    def collect_audio_transcripts(self):
        raise NotImplementedError
    
    def clear_audio_trans(self):
        """
        this method needs to be called between iterations of train/dev/test sets
        otherwise, the samples will accumulate with sucessive iteration calls
        """
        self.audio_trans = list()

    def write_json(self, save_path:str):
        """
        this method converts the audio files to wav format, filters out the 
        audio files based on the min and max duration and saves the audio_path, 
        transcript, and duration into a json file specified in the input save_path
        """
        # filter the entries by the duration bounds and write file
        unknown_words = UnknownWords()
        count_outside_duration = 0.0  # count the utterance outside duration bounds
        with open(save_path, 'w') as fid:
            logging.info("Writing files to label json")
            for audio_path, transcript in tqdm.tqdm(self.audio_trans):
                # skip the audio file if it doesn't exist
                if not os.path.exists(audio_path):
                    logging.info(f"file {audio_path} does not exists")
                    continue
                
                base, raw_ext = os.path.splitext(audio_path)
                # sometimes using the ".wv" extension so that original .wav files can be converted
                wav_path = base + os.path.extsep + self.audio_ext
                # if the wave file doesn't exist or it should be re-converted, convert to wave
                if not os.path.exists(wav_path) or self.force_convert:
                    try:
                        convert.to_wave(audio_path, wav_path)
                    except subprocess.CalledProcessError:
                        # if the file can't be converted, skip the file by continuing
                        logging.info(f"Process Error converting file: {audio_path}")
                        continue
                
                dur = wave.wav_duration(wav_path)
                if self.min_duration <= dur <= self.max_duration:
                    text = self.text_to_phonemes(transcript, unknown_words, wav_path, self.lex_dict)
                    # if transcript has an unknown word, skip it
                    if unknown_words.has_unknown: 
                        continue
                    
                    datum = {'text' : text,
                            'duration' : dur,
                            'audio' : wav_path}
                    json.dump(datum, fid)
                    fid.write("\n")
                else:
                    count_outside_duration += 1

        print(f"Count excluded because of duration bounds: {count_outside_duration}")
        unknown_words.process_save(save_path)


    def text_to_phonemes(self, transcript:str, unknown_words, audio_path:str, lex_dict:dict=None,):
        """
        this method removed unwanted puncutation marks split the text into a list of words
        or list of phonemes if a lexicon_dict exists
        """
        # allows for alphanumeric characters, space, and apostrophe
        accepted_char = '[^A-Za-z0-9 \']+'
        # filters out unaccepted characters, lowers the case
        try:
            transcript = re.sub(accepted_char, '', transcript).lower()
        except TypeError:
            logging.info(f"Type Error with: {transcript}")
        # check that all punctuation (minus apostrophe) has been removed 
        punct_noapost = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
        for punc in punct_noapost:
            if punc in transcript: 
                raise ValueError(f"unwanted punctuation {punc} in transcript")
        # split the transcript into a list of words
        transcript = transcript.split()
        # if there is a pronunciation dict, convert words to phonemes
        if self.lex_dict is not None:
            unknown_words.check_transcript(audio_path, transcript, self.lex_dict)
            phonemes = []
            for word in transcript:
                # TODO: I shouldn't need to include list() in get but dict is outputing None not []
                phonemes.extend(self.lex_dict.get(word, list()))
            transcript = phonemes

        return transcript


    def write_json_mp(self, data_json_path:str):
        """
        this method converts the audio files to wav format, filters out the 
        audio files based on the min and max duration and saves the audio_path, 
        transcript, and duration into a json file specified in the input save_path
        """
        NUM_PROC = mp.cpu_count()

        # clearing the data_json_path file contents so that values are appended
        # to older values from previous runs
        with open(data_json_path, 'w') as fid:
            fid.write('')

        # set the arguments not provided by self.audio_trans
        pool_fn = partial(
            self._process_sample, 
            data_json_path = data_json_path,
            force_convert = self.force_convert,
            min_duration = self.min_duration,
            max_duration = self.max_duration,
            lex_dict = self.lex_dict,
            download_audio = self.download_audio
        )
        # call the multi-process pool
        # the audio_trans list is broken into chunks to see the tqdm progress bar 
       
        chunk_size = mp.cpu_count() * 1000
        iterations = math.ceil(len(self.audio_trans) / chunk_size) 
        list_unk_dict = list()
        with mp. Pool(processes=NUM_PROC) as pool:
            for chunk_idx in tqdm.tqdm(range(iterations)):
           
                unk_word_dict = pool.map(
                    pool_fn, self.audio_trans[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
                )
                list_unk_dict.extend(unk_word_dict)

            pool.close()
            pool.join()
        
        print("finished worker pool")

        # combine the unk_word_dicts
        all_unk_dict = dict()
        for unk_word_dict in list_unk_dict:
            for word in unk_word_dict:
                all_unk_dict[word] = all_unk_dict.get(word, 0) + unk_word_dict[word]
        
        # write the unk_word_dict
        data_dir = os.path.split(data_json_path)[0]
        unk_words_filename = "unk-words-dict_{}.json".format(str(date.today()))
        unk_words_filename = os.path.join(data_dir, unk_words_filename)
        with open(unk_words_filename, 'w') as fid:
            json.dump(all_unk_dict, fid)

    def _process_sample(self,
                        audio_transcript:Tuple[str, str], 
                        data_json_path:str,
                        force_convert:bool,
                        min_duration:float, 
                        max_duration:float,
                        lex_dict:dict,
                        download_audio:bool=False) -> dict:
        """
        Unfortunately, there are four ways this function can exit and return the unk_word_dict.
        (1) if the audio_path doesn't exists, (2) if the convert.to_wave call fails, 
        (3) if the unk_word_dict returned by self.process_text_mp is non-empty,
        (4) if the unk_word_dict is empty. 
        
        The (4) path represents a successful function call, and the other three are provisions
        for undesirable outcomes. In the single-process version, the first three return statements
        were `continue` statements, which are now not applicable in a multi-process function.
        
        Args:
            audio_transcript: Tuple of audio file and transcript, positional arg in multi-processing 
            save_path: see argsparse description,  fixed arg using functools.partial
            force_convert: see argsparse description,  fixed arg using functools.partial
            min_duration: see argsparse description,  fixed arg using functools.partial
            max_duration: see argsparse description,  fixed arg using functools.partial
            lex_dict: see argsparse description,  fixed arg using functools.partial
            download_audio (bool): if true, the function will download the audio
        Returns:
            unk_word_dict: a dictionary of unknown with words as keys and counts as values
        """
        unk_words_dict = {}

        # unpack the transcript differently if audio will be downloaded
        if download_audio:
            (wav_path, download_url), transcript = audio_transcript

            # I don't want to use a context manager, so am using the old open/.close() notation
            tempfile = NamedTemporaryFile(suffix=".m4a")
            audio_path = tempfile.name
            # download the audio file into the tempfile
            try:
                urllib.request.urlretrieve(download_url, filename=audio_path)
            except (ValueError, urllib.error.URLError) as e:
                print(f"unable to download url: {download_url} due to exception: {e}")
                return unk_words_dict

        else:   # if not downloading, unpack and check if the path exists
            audio_path, transcript = audio_transcript
            # skip the audio file if it doesn't exist
            if not os.path.exists(audio_path):
                print(f"file {audio_path} does not exists")
                return unk_words_dict
            
            # replace the original extension with ".wav"
            wav_path = os.path.splitext(audio_path)[0] + os.path.extsep + "wav"
        

        # if the wave file doesn't exist, convert to wave
        #if not os.path.exists(wav_path) or force_convert:
        try:
            convert.to_wave(audio_path, wav_path)
        except subprocess.CalledProcessError:
            # if the file can't be converted, skip the file by continuing
            print(f"Process Error converting file: {audio_path}")
            return unk_words_dict
        
        # close the tempfile that contains the downloaded audio
        if download_audio:
            tempfile.close()    

        # filter by duration
        dur = wave.wav_duration(wav_path)
        if min_duration <= dur <= max_duration:

            text, unk_word_dict = self.text_to_phonemes_mp(transcript, lex_dict)
            # if transcript has an unknown word, skip it
            if unk_word_dict:
                return unk_word_dict
            else: 
                # write the datum and return an empty unk_word_dict
                with open(data_json_path, 'a+') as fid:
                    datum = {
                        'text' : text,
                        'duration' : dur,
                        'audio' : wav_path
                    }
                    json.dump(datum, fid)
                    fid.write("\n")

                return unk_word_dict


    def text_to_phonemes_mp(self, transcript:str, lex_dict:dict=None):
        """this method removed unwanted puncutation marks split the text into a list of words
        or list of phonemes if a lexicon_dict exists
        """
        # allows for alphanumeric characters, space, and apostrophe
        accepted_char = '[^A-Za-z0-9 \']+'

        # filters out unaccepted characters, lowers the case
        try:
            transcript = re.sub(accepted_char, '', transcript).lower()
        except TypeError:
            logging.info(f"Type Error with: {transcript}")

        # check that all punctuation (minus apostrophe) has been removed 
        punct_noapost = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
        for punc in punct_noapost:
            if punc in transcript:
                raise ValueError(f"unwanted punctuation {punc} in transcript")

        # split the transcript into a list of words
        transcript = transcript.split()
        # if there is a pronunciation dict, convert words to phonemes
        if lex_dict is not None:
            unk_words = [word for word in transcript if word not in lex_dict]
            unk_word_dict = Counter(unk_words)
            
            if unk_word_dict:
                transcript = []
            else:
                phonemes = []
                for word in transcript:
                    # TODO: I shouldn't need to include list() in get but dict is outputing None not []
                    phonemes.extend(lex_dict.get(word, list()))
                transcript = phonemes

        return transcript, unk_word_dict


###################   COMMON VOICE       #######################

class CommonvoicePreprocessor(DataPreprocessor):
    def __init__(self, dataset_dir, dataset_files, dataset_name, lexicon_path,
                        force_convert, min_duration, max_duration):
        super(CommonvoicePreprocessor, self).__init__(dataset_dir, dataset_files,
                                                      dataset_name, lexicon_path,
                                                      force_convert, min_duration, max_duration)

    def process_datasets(self):
        for set_name, label_name in self.dataset_dict.items():
            self.clear_audio_trans()    # clears the audio_transcript buffer
            label_path = os.path.join(self.dataset_dir, label_name)
            logging.info(f"label_path: {label_path}")
            self.collect_audio_transcripts(label_path)
            logging.info(f"len of auddio_trans: {len(self.audio_trans)}")
            root, ext = os.path.splitext(label_path)
            json_path = root + os.path.extsep + "json"
            logging.info(f"entering write_json for {set_name}")
            self.write_json(json_path)
        unique_unknown_words(self.dataset_dir)

    def collect_audio_transcripts(self, label_path:str):
        
        # open the file and select only entries with desired accents
        accents = ['us', 'canada']
        logging.info(f"Filtering files by accents: {accents}")
        dir_path = os.path.dirname(label_path)
        with open(label_path) as fid: 
            reader = csv.reader(fid, delimiter='\t')
            # first line in reader is the header which equals:
            # ['client_id', 'path', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent'] or 
            # ['client_id', 'path', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'vote_diff']
            header = next(reader)
            for line in reader:
                # filter by accent
                if line[7] in accents:
                    audio_path = os.path.join(dir_path, "clips", line[1])
                    transcript = line[2]
                    self.audio_trans.append((audio_path, transcript))


###################     TEDLIUM       ###################### 

class TedliumPreprocessor(DataPreprocessor):
    def __init__(self, dataset_dir, dataset_files, dataset_name, lexicon_path,
                        force_convert, min_duration, max_duration):
        super(TedliumPreprocessor, self).__init__(dataset_dir, dataset_files, dataset_name, 
                                    lexicon_path, force_convert, min_duration, max_duration)

        # legacy means the data are in the format of previous version. 
        # legacy contains all the data in tedlium v3
        #train_dir = os.path.join("legacy", "train")
        #dev_dir = os.path.join("legacy", "dev")
        #test_dir = os.path.join("legacy", "test")
        #self.dataset_dict = {"train":train_dir, "dev": dev_dir, "test": test_dir}


    def process_datasets(self):
        for set_name, label_name in self.dataset_dict.items():
            self.clear_audio_trans()    # clears the audio_transcript buffer
            data_path = os.path.join(self.dataset_dir, label_name)
            self.collect_audio_transcripts(data_path)
            json_path = os.path.join(self.dataset_dir, "{}.json".format(set_name))
            self.write_json(json_path)
        unique_unknown_words(self.dataset_dir)
    
    def collect_audio_transcripts(self, data_path:str):
        """
        """
        # create directory to store converted wav files 
        converted_dir = os.path.join(data_path, "converted")
        wav_dir = os.path.join(converted_dir, "wav")
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)

        ted_talks = os.listdir(os.path.join(data_path, "sph"))

        for sph_file in tqdm.tqdm(ted_talks, total=len(ted_talks)):
            
            speaker_name = os.path.splitext(sph_file)[0]
            sph_file_full = os.path.join(data_path, "sph", sph_file)
            stm_file_full = os.path.join(data_path, "stm", "{}.stm".format(speaker_name))
            assert os.path.exists(sph_file_full) and os.path.exists(stm_file_full),\
                f"source files {sph_file_full}, {stm_file_full} don't exist"
            
            all_utterances = self.get_utterances_from_stm(stm_file_full)
            all_utterances = filter(self.filter_utterances, all_utterances)
            
            for utterance_id, utterance in enumerate(all_utterances):
                target_fn = "{}_{}.wav".format(utterance["filename"], str(utterance_id))
                target_wav_file = os.path.join(wav_dir, target_fn)
                if not os.path.exists(target_wav_file) or self.force_convert:
                    # segment the ted_talks into individual utterances
                    try:
                        # cuts and writes the utterance
                        self.cut_utterance(sph_file_full, target_wav_file, 
                            utterance["start_time"], utterance["end_time"])
                    except: 
                        logging.info(f"Error in cutting utterance: {target_wav_file}")
                
                # audio_path is corrupted and is skipped
                if skip_file("tedlium", target_wav_file):
                    continue
                
                transcript = self.remove_unk_token(utterance["transcript"])
                audio_path = target_wav_file
                self.audio_trans.append((audio_path, transcript))

    def remove_unk_token(self, transcript:str):
        """
        removes the <unk> token from the transcript
        """
        unk_token = "<unk>"
        return transcript.replace(unk_token, "").strip()


    def get_utterances_from_stm(self, stm_file:str):
        """
        Return list of entries containing phrase and its start/end timings
        :param stm_file:
        :return:
        """
        res = []
        with io.open(stm_file, "r", encoding='utf-8') as f:
            for stm_line in f:
                tokens = stm_line.split()
                start_time = float(tokens[3])
                end_time = float(tokens[4])
                filename = tokens[0]
                transcript = unicodedata.normalize("NFKD",
                                                " ".join(t for t in tokens[6:]).strip()). \
                    encode("utf-8", "ignore").decode("utf-8", "ignore")
                if transcript != "ignore_time_segment_in_scoring":
                    res.append({
                        "start_time": start_time, "end_time": end_time,
                        "filename": filename, "transcript": transcript
                    })
                
            return res

    def filter_utterances(self, utterance_info, min_duration=1.0, max_duration=20.0)->bool:
        if (utterance_info["end_time"] - utterance_info["start_time"]) > min_duration:
            if (utterance_info["end_time"] - utterance_info["start_time"]) < max_duration:
                return True
        return False

    def cut_utterance(self, src_sph_file, target_wav_file, start_time, end_time, sample_rate=16000):
        cmd="sox {}  -r {} -b 16 -c 1 {} trim {} ={}".\
            format(src_sph_file, str(sample_rate), target_wav_file, start_time, end_time)
        subprocess.call([cmd], shell=True)



###################     VOXFORGE       ######################  
         
class VoxforgePreprocessor(DataPreprocessor):
    def __init__(self, dataset_dir, dataset_files, dataset_name, lexicon_path,
                        force_convert, min_duration, max_duration):
        super(VoxforgePreprocessor, self).__init__(dataset_dir, dataset_files, dataset_name, 
                                    lexicon_path, force_convert, min_duration, max_duration)
        
        #self.dataset_dict = {"all":"archive"}

    def process_datasets(self):
        for set_name, label_name in self.dataset_dict.items():
            self.clear_audio_trans()    # clears the audio_transcript buffer
            data_path = os.path.join(self.dataset_dir, label_name)
            self.collect_audio_transcripts(data_path)
            json_path = os.path.join(self.dataset_dir, "all.json")
            self.write_json(json_path)
        unique_unknown_words(self.dataset_dir)

    def collect_audio_transcripts(self, data_path:str):
        """
        Voxforge audio in "archive/<sample_dir>/wav/<sample_name>.wav" and
        transcripts are in the file "archive/sample_dir/etc/prompts-original"
        """
        audio_pattern = "*"
        pattern_path = os.path.join(data_path, audio_pattern)
        list_sample_dirs = glob.glob(pattern_path)
        possible_text_fns = ["prompts-original", "PROMPTS", "Transcriptions.txt",  
                                "prompt.txt", "prompts.txt", "therainbowpassage.prompt", 
                                "cc.prompts", "a13.text"]
        logging.info("Processing the dataset directories...")
        for sample_dir in tqdm.tqdm(list_sample_dirs):
            text_dir = os.path.join(sample_dir, "etc")
            # find the frist filename that exists in the directory
            for text_fn in possible_text_fns:
                text_path = os.path.join(text_dir, text_fn)
                if os.path.exists(text_path):
                    break
            with open(text_path, 'r') as fid:
                for line in fid:
                    line = line.strip().split()
                    # if an empty entry, skip it
                    if len(line)==0:
                        continue 
                    audio_name = self.parse_audio_name(line[0])
                    audio_path = self.find_audio_path(sample_dir, audio_name)
                    if audio_path is None:
                        continue
                    # audio_path is corrupted and is skipped
                    elif skip_file(audio_path):
                        continue
                    transcript = line[1:]
                    # transcript should be a string
                    transcript = " ".join(transcript)
                    self.audio_trans.append((audio_path, transcript))

    def parse_audio_name(self, raw_name:str)->str:
        """
        Extracts the audio_name from the raw_name in the PROMPTS file.
        The audio_name should be the last separate string.
        """
        split_chars = r'[/]'
        return re.split(split_chars, raw_name)[-1]


    def find_audio_path(self, sample_dir:str, audio_name:str)->str:
        """
        Most of the audio files are in a dir called "wav" but
        some are in the "flac" dir with the .flac extension
        """
        possible_exts =["wav", "flac"]
        found = False
        for ext in possible_exts:
            file_name = audio_name + os.path.extsep + ext
            audio_path = os.path.join(sample_dir, ext, file_name)
            if os.path.exists(audio_path):
                found =  True
                break
        if not found: 
            audio_path = None
            logging.info(f"dir: {sample_dir} and name: {audio_name} not found")
        return audio_path


###################   TATOEBA       ######################  

class TatoebaPreprocessor(DataPreprocessor):
    def __init__(self, dataset_dir, dataset_files, dataset_name, lexicon_path,
                        force_convert, min_duration, max_duration):
        super(TatoebaPreprocessor, self).__init__(dataset_dir, dataset_files, dataset_name, 
                                    lexicon_path, force_convert, min_duration, max_duration)
        #self.dataset_dict = {"all":"sentences_with_audio.csv"}

    def process_datasets(self):
        logging.info("In Tatoeba process_datasets")
        for set_name, label_fn in self.dataset_dict.items():
            self.clear_audio_trans()    # clears the audio_transcript buffer
            label_path = os.path.join(self.dataset_dir, label_fn)
            self.collect_audio_transcripts(label_path)
            root, ext = os.path.splitext(label_path)
            json_path = root + os.path.extsep + "json"
            self.write_json(json_path)
        unique_unknown_words(self.dataset_dir)
    

    def collect_audio_transcripts(self, label_path:str):
        # open the file and select only entries with desired accents
        speakers = ["CK", "Delian", "pencil", "Susan1430"]  # these speakers have north american accents
        logging.info(f"Filtering files by speakers: {speakers}")
        error_files = {"CK": {"min":6122903, "max": 6123834}} # files in this range are often corrupted
        dir_path = os.path.dirname(label_path)
        with open(label_path) as fid: 
            reader = csv.reader(fid, delimiter='\t')
            # first line in reader is the header which equals:
            # ['id', 'username', 'text']
            is_header = True
            for line in reader:
                if is_header:
                    is_header=False
                    continue
                else: 
                    # filter by accent
                    if line[1] in speakers:
                        audio_path = os.path.join(dir_path, "audio", line[1], line[0]+".mp3")
                        transcript = " ".join(line[2:])
                        if skip_file(audio_path):
                            logging.info(f"skipping {audio_path}")
                            continue
                        self.audio_trans.append((audio_path, transcript))


###################   SPEAK TRAIN       ###################### 


class SpeakTrainPreprocessor(DataPreprocessor):
    def __init__(self, dataset_dir, dataset_files, dataset_name, lexicon_path,
                        force_convert, min_duration, max_duration, *args):
        super(SpeakTrainPreprocessor, self).__init__(dataset_dir, dataset_files,
                                                      dataset_name, lexicon_path,
                                                      force_convert, min_duration, max_duration)


    def process_datasets(self):
        for set_name, label_name in self.dataset_dict.items():
            # clears the audio_transcript buffer
            self.clear_audio_trans()    

            label_path = os.path.join(self.dataset_dir, label_name)
            logging.info(f"label_path: {label_path}")

            self.collect_audio_transcripts(label_path)
            
            logging.info(f"len of auddio_trans: {len(self.audio_trans)}")
            root, ext = os.path.splitext(label_path)
            json_path = root + os.path.extsep + "json"
            logging.info(f"entering write_json for {set_name}. writing json to {json_path}")
            self.write_json(json_path)


    def collect_audio_transcripts(self, label_path:str):
        
        audio_dir = os.path.join(
            os.path.split(label_path)[0], "audio"
        )
        audio_ext = "m4a"
        speaker_counter = dict()

        with open(label_path, 'r') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            # header: id, text, lessonId, lineId, uid, date
            header = next(tsv_reader)
            
            for row in tqdm.tqdm(tsv_reader):                
                audio_path = os.path.join(audio_dir, row[0] + os.extsep + audio_ext)
                
                # skip the file if it is in one of the speak test sets
                if skip_file(audio_path, "speaktrain"):
                    continue
                
                # no longer need to check if path exists when using trimmed_data.tsv
                # the `elif` call below makes the script very slow as it is IO (disk) limited
                #elif not os.path.exists(audio_path):
                #    continue
                
                else:
                    self.audio_trans.append((audio_path, row[1]))


###################   SPEAK TRAIN       ###################### 
###############    FROM FULL METDATA.TSV   ###################


class SpeakTrainMetadataPreprocessor(DataPreprocessor):
    """This class filters, downloads, and preprocesses speak training data
    based on the metadata from all (as of 2020-12-28) the recordings that meet
    the `target==guess` criterion. 

    This class is filtering the recordings in the metadata.tsv file based on the count constraints
    on speaker and target-sentence in the config file


    """
    def __init__(self, config:dict):
        """Takes in a `config` dictionary as input
        """
        super(SpeakTrainMetadataPreprocessor, self).__init__(
            dataset_dir = config['dataset_dir'], 
            dataset_files = config['dataset_files'],
            dataset_name = config['dataset_name'],
            lexicon_path = config['lexicon_path'],
            force_convert = config['force_convert'],
            min_duration = config['min_duration'],
            max_duration = config['max_duration'],
            download_audio = config['download_audio']
        )
        self.config = config


    def process_datasets(self):
        """Main function that knits together supporting functions
        """
        # iterate through every dataset in the dataset_dict
        for name, label_path in self.dataset_dict.items():
            # clears the audio_transcript buffer
            self.clear_audio_trans()    

            label_path = os.path.join(self.dataset_dir, label_path)
            logging.info(f"label_path: {label_path}")
            # collects the audio paths and transcripts into a list
            self.collect_audio_transcripts(label_path)
            
            logging.info(f"len of auddio_trans: {len(self.audio_trans)}")

            # creates the audio path name
            json_path = os.path.join(self.dataset_dir, name + ".json")
            logging.info(f"entering write_json for {name}. writing json to {json_path}")

            self.write_json_mp(json_path)


    def collect_audio_transcripts(self, metadata_path:str):
        
        audio_dir = os.path.join(self.dataset_dir, "audio")
        audio_ext = "wav"
        speaker_counter = dict()
        
        # creates mapping from record_id to other ids like lesson, speaker, and target sentence
        disjoint_ids_map = get_record_ids_map(self.config['disjoint_metadata'], has_url=False)

        disjoint_id_sets  = get_disjoint_sets(self.config['disjoint_datasets'], disjoint_ids_map)
        
        del disjoint_ids_map        # to conserve memory
    
        count_constraints = {
            name: int(value * self.config['dataset_size']) 
            for name, value in config['constraints'].items()
        }
        id_counter = {name: dict() for name in count_constraints}

        examples_collected = 0
        with open(metadata_path, 'r') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            # header: id, target, lessonId, lineId, uid, redWords, date, audio_url
            header = next(tsv_reader)
            
            while examples_collected < self.config['dataset_size']:

                # print progress of collecting examples
                if examples_collected != 0 and examples_collected % config['print_modulus'] == 0:
                    print(f"{examples_collected} examples collected")
                try:
                    row = next(tsv_reader)
                except StopIteration:
                    print(f"Stop encountered {examples_collected} examples collected")
                    break

                assert len(row) == 8, f"metadata row length: {len(row)} is not 8"

                record_id = row[0]
                # creates a mapping from record_id to other id's for filtering and contraints
                record_ids_map = {
                    record_id: {
                        'record': record_id,                    # record
                        'target_sentence': process_text(row[1]), # processed target
                        'lesson': row[2],                       # lesson
                        'speaker': row[4]                       # speaker
                    }
                }
                # check if the record_id should be disjoint from the disjoint_sets
                pass_filter = check_disjoint_filter(record_id, disjoint_id_sets, record_ids_map)
                if pass_filter:
                    # check if the count for each id-type is under the contraint limit
                    pass_constraints = check_update_contraints(
                        record_id, 
                        record_ids_map, 
                        id_counter, 
                        count_constraints
                    )
                    if pass_constraints:
                        # add the example to the list to be written
                        audio_path = os.path.join(audio_dir, row[0] + os.extsep + audio_ext)
                        # a tuple of the audio path and download url allow for the downloading 
                        # of the audio file in the `write_json` function
                        self.audio_trans.append(( 
                            (audio_path, row[7]), row[1]
                        ))
                        examples_collected += 1


class UnknownWords():

    def __init__(self):
        self.word_set:set = set()
        self.filename_dict:dict = dict()
        self.line_count:int = 0
        self.word_count:int = 0
        self.has_unknown= False

    def check_transcript(self, filename:str, text:str, word_phoneme_dict:dict):
        
        # convert the text into a list if it is a string
        if type(text) == str: 
            text = text.split()
        elif type(text) == list: 
            pass
        else: 
            raise(TypeError("input text is not string or list type"))

        # increment the line and word counts
        self.line_count += 1
        self.word_count += len(text) - 1
        
        # if the word_phoneme_dict doesn't have an entry for 'word', it is an unknown word
        line_unk = [
            word for word in text 
            if not word_phoneme_dict.get(word, data_helpers.UNK_WORD_TOKEN)
        ]
        
        #if line_unk is empty, has_unknown is False
        self.has_unknown = bool(line_unk)

        # if unknown words exist, update the word_set and log the count per filename
        if self.has_unknown:
            self.word_set.update(line_unk)
            self.filename_dict.update({filename: len(line_unk)})

    def process_save(self, label_path:str)->None:
        """
        saves a json object of the dictionary with relevant statistics on the unknown words in corpus
        """
        stats_dict=dict()
        stats_dict.update({"unique_unknown_words": len(self.word_set),
                            "count_unknown_words": sum(self.filename_dict.values()),
                            "total_words": self.word_count,
                            "lines_unknown_words": len(self.filename_dict),
                            "total_lines": self.line_count,
                            "unknown_words_set": list(self.word_set),
                            "unknown_words_dict": self.filename_dict})
        
        dir_path, base_ext = os.path.split(label_path)
        base, ext = os.path.splitext(base_ext)
        stats_dir = os.path.join(dir_path, "unk_word_stats")
        os.makedirs(stats_dir, exist_ok=True)
        unk_words_filename = "{}_unk-words-stats_{}.json".format(base, str(date.today()))
        stats_dict_fn = os.path.join(stats_dir, unk_words_filename)
        with open(stats_dict_fn, 'w') as fid:
            json.dump(stats_dict, fid)
        

def unique_unknown_words(dataset_dir:str):
    """ 
    Creates a set of the total number of unknown words across all segments in a dataset assuming a
    unk-words-stats.json file from process_unknown_words() has been created for each part of the dataset. 
    Arguments:
        dataset_dir (str): pathname of dir continaing "unknown_word_stats" dir with unk-words-stats.json files
    """
    pattern = os.path.join(dataset_dir, "unk_word_stats", "*unk-words-stats*.json")
    dataset_list = glob.glob(pattern)
    unknown_set = set()
    for data_fn in dataset_list: 
        with open(data_fn, 'r') as fid: 
            unk_words_dict = json.load(fid)
            unknown_set.update(unk_words_dict['unknown_words_set'])
            logging.info(f"for {data_fn}, # unique unknownw words: {len(unk_words_dict['unknown_words_set'])}")

    unknown_set = filter_set(unknown_set)
    unknown_list = list(unknown_set)
    filename = "all_unk_words_{}.txt".format(str(date.today()))
    write_path = os.path.join(dataset_dir, "unk_word_stats", filename)
    with open(write_path, 'w') as fid:
        fid.write('\n'.join(unknown_list))
    logging.info(f"number of filtered unknown words: {len(unknown_list)}")


def filter_set(unknown_set:set):
    """
    currently no filtering is being done. Previously, it had filters the set based on the length 
    and presence of digits.
    """
    # unk_filter = filter(lambda x: len(x)<30, unknown_set)
    # search_pattern = r'[0-9!#$%&()*+,\-./:;<=>?@\[\\\]^_{|}~]'
    # unknown_set = set(filter(lambda x: not re.search(search_pattern, x), unk_filter))
    return unknown_set


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="creates a data json file")
    parser.add_argument(
        "--config", type=str, help="path to preprocessing config."
    )
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file) 

    start_time = time.time()
    data_preprocessor = eval(config['dataset_name']+"Preprocessor")
    data_preprocessor = data_preprocessor(config) 
    data_preprocessor.process_datasets()
    print(f"script took: {round((time.time() - start_time)/ 60, 3)} min")
