"""
this file contains functions that allow for audio files to be fed to Google's
speech-to-text API. This will be used to see if the STT can identify misprounced words.
Copyright: Speak Labs 2021
Author: Dustin Zubke
"""
# standard libs
import argparse
import random
from pathlib import Path
import os
# third-party libs
from google.cloud import speech_v1p1beta1 as speech
# project libs
from speech.utils.data_helpers import get_record_ids_map, path_to_id, process_text
from speech.utils.io import read_data_json


# flag transcripts with more than 1 alternatives
# group by the apple and google transcripts being the same and differnt

def stt_on_sample(data_path:str, metadata_path:str, save_path:str)->None:
    """Pulls a random sample of audio files from `data_path` and calls
    Google's speech-to-text API to get transcript predictions. The Google STT
    output is formated and written to `save_path` along with the files's transcript
    from `metadata_path`. 

    Args:
        data_path: path to training json 
        metadata_path: path to metadata tsv containing transcript
        save_path: path where output txt will be saved
    """
    random.seed(0)
    SAMPLE_SIZE = 100

    data = read_data_json(data_path)
    data_sample = random.choices(data, k=SAMPLE_SIZE)
    print(f"sampling {len(data_sample)} samples from {data_path}")

    # mapping from audio_id to transcript
    metadata = get_record_ids_map(metadata_path, has_url=True)

    client = speech.SpeechClient()
    
    preds_with_two_trans = set()
    match_trans_entries = list()       # output list for matching transcripts
    diff_trans_entries = list()        # output list for non-matching transcripts
    for datum in data_sample:
        audio_path = datum['audio']
        audio_id = path_to_id(audio_path)
        id_plus_dir = os.path.join(*audio_path.split('/')[-2:])

        # format the input and submit a call to the STT API
        with open(audio_path, "rb") as fid:
            content = fid.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_word_confidence=True,
        )
        response = client.recognize(config=config, audio=audio)
        
        for result in response.results:
            if len(result.alternatives) > 1:
                preds_with_two_trans.add(audio_id)
            alt = result.alternatives[0]
    
            id_entry = formatted_entry(alt, metadata[audio_id]['target_sentence'], id_plus_dir)

            apl_trans = process_text(metadata[audio_id]['target_sentence'])
            ggl_trans = process_text(alt.transcript)

            if apl_trans == ggl_trans:
                match_trans_entries.append(id_entry)
            else:
                diff_trans_entries.append(id_entry)

    print(f"number of audio with two predictions: {len(preds_with_two_trans)}")
    print("ids: ")
    print("\n".join(preds_with_two_trans))   

    with open(save_path, 'w') as fid:
        for entries in [diff_trans_entries, match_trans_entries]:
            fid.write("-"*10+'\n')
            for entry in entries:
                fid.write(entry+'\n\n')
                
    

def formatted_entry(pred_alt, apl_trans:str, id_plus_dir:str)->str:
    """Formats the entry for each prediction 
    """
    lines = [
        id_plus_dir,
        apl_trans,
        pred_alt.transcript, 
        str(round(pred_alt.confidence, 4)),
        ' '.join([f"({x.word}, {str(round(x.confidence, 4))})" for x in pred_alt.words])
    ]
    
    return '\n'.join(lines)
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calls Google's speech-to-text api and writes results to json."
    )
    parser.add_argument(
        "--data-path", help="path to training json where audio examples will be samples from"
    )
    parser.add_argument(
        "--metadata-path", help="path to metadata file containing transcript"
    )
    parser.add_argument(
        "--save-path", help="path where output file will be saved"
    )
    
    args = parser.parse_args()

    stt_on_sample(args.data_path, args.metadata_path, args.save_path)



