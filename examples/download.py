# standard libraries
import argparse
import csv
import glob
import os
import re
from shutil import copyfile
import tarfile
import urllib.request
from zipfile import ZipFile
# third party libraries
import tqdm
# project libraries
from speech.utils.convert import to_wave



class Downloader(object):

    def __init__(self, output_dir, dataset_name):
        self.output_dir = output_dir
        self.dataset_name = dataset_name.lower()
        self.download_dict = dict()
        self.ext = ".tar.gz"


    def download_dataset(self):
        save_dir = self.download_extract()
        print(f"Data saved and extracted to: {save_dir}")
        self.extract_samples(save_dir)

    def download_extract(self):
        """
        Standards method to download and extract zip file
        """
        save_dir = os.path.join(self.output_dir, self.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        for name, url in self.download_dict.items():
            if name == "data":
                if os.path.exists(os.path.join(save_dir, self.data_dirname)):
                    print("Skipping data download")
                    continue
    
            save_path = os.path.join(save_dir, name + self.ext)
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            with tarfile.open(save_path) as tf:
                tf.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir
    
    def extract_samples(self, save_dir:str):
        """
        Most datasets won't need to extract samples
        """
        pass
    

class VoxforgeDownloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        super(VoxforgeDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {
            "data": "https://s3.us-east-2.amazonaws.com/common-voice-data-download/voxforge_corpus_v1.0.0.tar.gz",
            "lexicon": "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Lexicon/VoxForge.tgz"
        }
        self.data_dirname = "archive"


    def extract_samples(save_dir:str):
        """
        All samples are zipped in their own tar files.
        This function collects all the tar filenames and
        unzips themm.
        """
        pattern = "*.tgz"
        sample_dir = os.path.join(save_dir, self.data_dirname)
        tar_path = os.path.join(sample_dir, pattern)
        tar_files = glob.glob(tar_path)
        print("Extracting and removing sample files...")
        for tar_file in tqdm(tar_files):
            with tarfile.open(tar_file) as tf:
                tf.extractall(path=sample_dir)
            os.remove(tar_file)

class TatoebaDownloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        super(TatoebaDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = { 
            "data": "https://downloads.tatoeba.org/audio/tatoeba_audio_eng.zip"
        }
        self.data_dirname = "audio"

        # downloads
        # https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
        # https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_CC0.tsv.bz2
        # https://downloads.tatoeba.org/exports/sentences_with_audio.tar.bz2
        # https://downloads.tatoeba.org/exports/user_languages.tar.bz2
        # https://downloads.tatoeba.org/exports/users_sentences.csv


    def download_extract(self):
        """
        Standards method to download and extract zip file
        """
        save_dir = os.path.join(self.output_dir,"commmon-voice")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, url in download_dict.items():
            if name == "data":
                if os.path.exists(os.path.join(save_dir, self.data_dirname)):
                    print("Skipping data download")
                    continue
            save_path = os.path.join(save_dir, name + ".tar.gz")
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            raise NotImplementedError("Zip can't yet be extracted with tarfile code")
            with tarfile.open(save_path) as tf:
                tf.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir


class TatoebaV2Downloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        super(TatoebaDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = { 
            "data": "",
            "data_csv": "https://downloads.tatoeba.org/exports/sentences_with_audio.tar.bz2"
        }
        self.data_dirname = "audio"


    def download_extract(self):
        """
        Standards method to download and extract zip file
        """
        save_dir = os.path.join(self.output_dir,"commmon-voice")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, url in download_dict.items():
            if name == "data":
                if os.path.exists(os.path.join(save_dir, self.data_dirname)):
                    print("Skipping data download")
                    continue
            save_path = os.path.join(save_dir, name + ".tar.gz")
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            raise NotImplementedError("Zip can't yet be extracted with tarfile code")
            with tarfile.open(save_path) as tf:
                tf.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir




class CommonvoiceDownloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        """
        A previous version of common voice (v4) can be downloaded here:
        "data":"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz"
        """
        super(CommonvoiceDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {
            "data":"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/en.tar.gz"
        }
        self.data_dirname = "clips"
    


class WikipediaDownloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        """
        """
        super(WikipediaDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {
            "data":"https://www2.informatik.uni-hamburg.de/nats/pub/SWC/SWC_English.tar"
        }
        self.data_dirname = "not-used"
        self.ext = ".tar"
    

class SpeakTrainDownloader(Downloader):
    
    def __init__(self, output_dir, dataset_name):
        """
        Downloading the Speak Train Data is significantly different than other datasets
        because the Speak data is not pre-packaged into a zip file. Therefore, the 
        `download_dataset` method will be overwritten. 
        """
        pass

    def download_dataset(self):
        """
        This method loops through the firestore document database using paginated queries based on
        the document id. It filters out documents where `target != guess` and saves the audio file
        and target text into separate files. 

        The approach to index the queries based on the document `id` is based on the approach
        outlined here: 
        https://firebase.google.com/docs/firestore/query-data/query-cursors#paginate_a_query
        """

        import firebase_admin
        from firebase_admin import credentials
        from firebase_admin import firestore
        import urllib.request

        PROJECT_ID = 'speak-v2-2a1f1'
        SAVE_DIR = "/home/dzubke/awni_speech/data/speak_train/data/"
        AUDIO_EXT = ".m4a"
        TXT_EXT = ".txt"
        QUERY_LIMIT = 1000    
        LAST_ID_PRINT_INTERVAL = 1000  # this should be a mutliple of the QUERY_LIMIT
     
        # verify and set the credientials
        CREDENTIAL_PATH = "/home/dzubke/awni_speech/speak-v2-2a1f1-d8fc553a3437.json"
        assert os.path.exists(CREDENTIAL_PATH), "Credential file does not exist or is in the wrong location."
        # set the enviroment variable that `firebase_admin.credentials` will use
        os.putenv("GOOGLE_APPLICATION_CREDENTIALS", CREDENTIAL_PATH)
         
        # ensure the save directory exits
        audio_dir = os.path.join(SAVE_DIR, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        example_data_path = os.path.join(SAVE_DIR, "train_data.tsv")
        with open(example_data_path, 'w', newline='\n') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            header = [
                "id", "text", "lessonId", "lineId", "uid", "date"
            ]
            tsv_writer.writerow(header)

        # initialize the credentials and firebase db client
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {'projectId': PROJECT_ID})
        db = firestore.client()

        # create the first query based on the constant QUERY_LIMIT
        rec_ref = db.collection(u'recordings')
        next_query = rec_ref.order_by(u'id').limit(QUERY_LIMIT)
        
        # dictionary to count the total and filtered audio files
        count_dict = {
            "total": 0,
            "filtered": 0
        }

        # loops until break is called in try-except block
        #while True:
        
        #testing the code with this loop
        loop_iterations = 3
        while loop_iterations > 0:
            # converting generator to list to it can be referenced mutliple times
            docs = list(next_query.stream())

            try:
                # this `id` will be used to start the next query
                last_id = docs[-1].to_dict()[u'id']
            # if docs is empty meaning there are no new documents, 
            # an IndexError will be raised and break the while loop
            except IndexError:
                break
            
            with open(example_data_path, 'a', newline='\n') as tsv_file:
                tsv_writer = csv.writer(tsv_file, delimiter='\t')
                
                # filter the documents where `target`==`guess`
                for doc in docs:
          
                    doc_dict = doc.to_dict() 
                    original_target = doc_dict['info']['target']
             
                    # some of the guess's don't include apostrophes
                    # so the filter criterion will not use apostrophes
                    target = self.process_text(doc_dict['info']['target'])
                    target_no_apostrophe = target.replace("'", "")
                    
                    guess = self.process_text(doc_dict['result']['guess'])
                    guess_no_apostrophe = guess.replace("'", "")

                    if target_no_apostrophe == guess_no_apostrophe:
                        # save the audio file from the link in the document
                        audio_url = doc_dict['result']['audioDownloadUrl']
                        audio_save_path = os.path.join(audio_dir, doc_dict['id'] + AUDIO_EXT)
                        urllib.request.urlretrieve(audio_url, filename=audio_save_path)
                        
                        # save the target in a tsv row
                        # tsv header: "id", "text", "lessonId", "lineId", "uid", "date"
                        tsv_row =[
                            doc_dict['id'], 
                            original_target, 
                            doc_dict['info']['lessonId'],
                            doc_dict['info']['lineId'],
                            doc_dict['user']['uid'],
                            doc_dict['info']['date']
                        ]
                        tsv_writer.writerow(tsv_row)
                        
                        count_dict['filtered'] += 1
                    
                    count_dict['total'] += 1
           
            # print the last_id every 5000 samples so the script can pick up from the last_id
            # if something breaks 
            if count_dict['total'] % LAST_ID_PRINT_INTERVAL == 0:
                print(f"last_id: {last_id} at count: {count_dict['total']}")
            
            # create the next query starting after the last_id 
            next_query = (
                rec_ref
                .order_by(u'id')
                .start_after({
                    u'id': last_id
                })
                .limit(QUERY_LIMIT))
            
            loop_iterations -= 1
        
        print(f"count_dict: {count_dict}")

    @staticmethod 
    def process_text(transcript:str):
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

class Chime1Downloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        super(Chime1Downloader, self).__init__(output_dir, dataset_name)
        # original dataset is in 2-channel. 
        # in this case, I have downloaded and transfered the data to the VM myself. 



class DemandDownloader(Downloader):

    def __init__(self, output_dir, dataset_name):
        """
        Limitations: 
            - feed_model_dir, the directory where the noise is fed to the model, is hard-coded
        """
        super(DemandDownloader, self).__init__(output_dir, dataset_name)
        self.download_dict = {}
        self.feed_model_dir = "/home/dzubke/awni_speech/data/noise/feed_to_model"
        self.load_download_dict()

    def load_download_dict(self):
        """
        Loads the download dictionary with download links and dataset names

        Assumptions: 
            Download links will still work in the future
        """

        noise_filenames = ["TMETRO_16k.zip", "TCAR_16k.zip", "SPSQUARE_16k.zip", "SCAFE_48k.zip",
                            "PSTATION_16k.zip", "PRESTO_16k.zip", "OMEETING_16k.zip", "OHALLWAY_16k.zip",
                            "NRIVER_16k.zip", "NPARK_16k.zip", "NFIELD_16k.zip", "DWASHING_16k.zip", 
                            "DLIVING_16k.zip", "DKITCHEN_16k.zip"]
        
        download_link = "https://zenodo.org/record/1227121/files/{}?download=1"
        
        for filename in noise_filenames:
            basename = os.path.splitext(filename)[0]
            self.download_dict.update({basename: download_link.format(filename)})


    def download_extract(self):
        """
        This method was overwritten because the download files were in zip format
        instead of tar format as in the base downloader class. 
        
        Limitations: 
            - this method is not idempotent as it will re-download and extract
                the files again if it is called again
        """
        save_dir = os.path.join(self.output_dir, self.dataset_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for name, url in self.download_dict.items():
            save_path = os.path.join(save_dir, name + ".zip")
            print(f"Downloading: {name}...")
            urllib.request.urlretrieve(url, filename=save_path)
            print(f"Extracting: {name}...")
            with ZipFile(save_path) as zipfile:
                zipfile.extractall(path=save_dir)
            os.remove(save_path)
            print(f"Processed: {name}")
        return save_dir


    def extract_samples(self, save_dir:str):
        """
        Extracts the wav files from the directories and copies them into the noise_dir.
        The audio files in the "SCAFE_48k" data subset are in 48 kHz and should be converted
        to 16 kHz. The if-statement in the for-loop does this conversion.

        Assumptions: 
            - The directory structure of the zip files will not change
            - 
        """
        pattern = "*/*.wav"
        high_res_audio = {"SCAFE"}
        all_wav_paths = glob.glob(os.path.join(save_dir, pattern))

        print("Extracting and removing sample files...")
        for wav_path in tqdm.tqdm(all_wav_paths):
            filename = os.path.basename(wav_path)
            dirname = os.path.basename(os.path.dirname(wav_path))
            dst_filename = "{}_{}".format(dirname, filename)
            dst_wav_path = os.path.join(self.feed_model_dir, dst_filename)
            if os.path.exists(dst_wav_path):
                print(f"{dst_wav_path} exists. Skipping...")
                continue
            else:
                # if the wavs are high resolution, down-convert to 16kHz
                if dirname in high_res_audio:
                    to_wave(wav_path, dst_wav_path)
                # if not high-res, just copy
                else: 
                    copyfile(wav_path, dst_wav_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Download voxforge dataset.")

    parser.add_argument("--output-dir",
        help="The dataset is saved in <output-dir>/<dataset-name>.")
    parser.add_argument("--dataset-name", type=str,
        help="Name of dataset with a capitalized first letter.")
    args = parser.parse_args()

    downloader = eval(args.dataset_name+"Downloader")(args.output_dir, args.dataset_name)
    print(f"type: {type(downloader)}")
    downloader.download_dataset()
