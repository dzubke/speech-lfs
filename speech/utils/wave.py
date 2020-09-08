# compatibility libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# standard libraries
import io
from tempfile import NamedTemporaryFile
# third-party libraries
from google.cloud import storage
from google.cloud.storage.blob import Blob      
import numpy as np
import soundfile

def array_from_wave(file_name:str, load_from_gcs:bool=False):
    """
    Args:
        file_name: filename to wav file on disk or in cloud storage. If cloud storage file, the file_name
            must start with the 'gs://' prefix.
        load_from_gcs: if true, the file_name will be treated as a path to a google cloud storage object
    """
    # reading file from cloud storage
    if load_from_gcs: 
        assert file_name.startswith("gs://"), f"filename {file_name} doesn't being with 'gs://'"
        client = storage.Client() 
        blob = Blob.from_string(file_name, client)
        with NamedTemporaryFile(suffix=".wav") as tmp_wav_file:
            wav_filename = tmp_wav_file.name
            with open(wav_filename, "wb") as file_obj:
                blob.download_to_file(file_obj)
            audio, samp_rate = soundfile.read(wav_filename, dtype='int16')
            return audio, samp_rate
    # reading file from disk
    else:
        audio, samp_rate = soundfile.read(file_name, dtype='int16')
        return audio, samp_rate

def str_load(path:str): 
    client = storage.Client() 
    blob = Blob.from_string(path, client) 
    file_as_string = blob.download_as_string() 
    data, sample_rate = soundfile.read(io.BytesIO(file_as_string), dtype='int16') 
    return data, sample_rate 



def wav_duration(file_name):
    audio, samp_rate = soundfile.read(file_name, dtype='int16')
    nframes = audio.shape[0]
    duration = nframes / samp_rate
    return duration
 
def array_to_wave(filename:str, audio_data:np.ndarray, samp_rate:int):
    """
    Writes an array to wave in the in the signed int-16 subtype (PCM_16)
    """
    soundfile.write(filename, audio_data, samp_rate, subtype='PCM_16', format='WAV')
