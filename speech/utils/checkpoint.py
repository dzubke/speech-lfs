"""
"""
import os
from google.cloud import storage

class GCSCheckpointHandler():
    def __init__(self, cfg):
        self.client = storage.Client()
        #self.local_save_file = hydra.utils.to_absolute_path(cfg.local_save_file)
        self.gcs_bucket = cfg['gcs_bucket']
        self.gcs_dir = cfg['gcs_dir']
        self.bucket = self.client.bucket(bucket_name=self.gcs_bucket)
        self.chkpt_per_epoch = cfg['checkpoints_per_epoch']


    def find_latest_checkpoint(self):
        """
        Finds the latest checkpoint in a folder based on the timestamp of the file.
        Downloads the GCS checkpoint to a local file, and returns the local file path.
        If there are no checkpoints, returns None.
        :return: The latest checkpoint path, or None if no checkpoints are found.
        """
        prefix = self.gcs_save_folder + self.prefix
        paths = list(self.client.list_blobs(self.gcs_bucket, prefix=prefix))
        if paths:
            paths.sort(key=lambda x: x.time_created)
            latest_blob = paths[-1]
            latest_blob.download_to_filename(self.local_save_file)
            return self.local_save_file
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
