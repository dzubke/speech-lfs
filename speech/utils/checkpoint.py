"""
"""
import os
from google.cloud import storage

class GCSCheckpointHandler():
    def __init__(self, cfg):
        self.client = storage.Client()
        self.local_save_dir = cfg['local_save_dir']
        self.gcs_bucket = cfg['gcs_bucket']
        self.gcs_dir = cfg['gcs_dir']
        self.bucket = self.client.bucket(bucket_name=self.gcs_bucket)
        self.chkpt_per_epoch = cfg['checkpoints_per_epoch']

    def find_gcs_object(self, filename:str):
        """
        Finds an object with `filename` in the gcs save directory. If it exists, the
        object is downloaded to a local file, and returns the local file path.
        If there are no checkpoints, returns None.
        :return: the local path to object or None if no objects are found.
        """
        prefix = self.gcs_dir + filename
        paths = list(self.client.list_blobs(self.gcs_bucket, prefix=prefix))
        if paths:
            #paths.sort(key=lambda x: x.time_created)
            #latest_blob = paths[-1]
            local_path = os.path.join(self.local_save_dir, filename)
            paths[0].download_to_filename(local_path)
            return local_path
        else:
            return None

    def _save_model(self, filepath: str, trainer, pl_module):

        # make paths
        if trainer.is_global_zero:
            tqdm.write("Saving model to %s" % filepath)
            trainer.save_checkpoint(filepath)
            self._save_file_to_gcs(filepath)

    def save_to_gcs(self, local_dir, file_name):
        gcs_path = os.path.join(self.gcs_dir, file_name)
        local_path = os.path.join(local_dir, file_name)
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
