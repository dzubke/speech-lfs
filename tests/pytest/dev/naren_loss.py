# standard libs
import os
# third-party libs
import numpy as np
# project libs
from speech.utils.io import read_pickle
from speech.utils import get_naren_model_optimizer


def naren_loss():
   
    file_dir =  os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(file_dir, "ctc_config_ph3.yaml")
    model, optimizer = get_naren_model_optimizer.get_model_optimizer(config_path)
    print(model)
    batch = get_saved_batch()

    loss = model.loss(batch)   
    print(f"naren loss: {loss}")
    print(f"naren loss item: {loss.item()}")

def get_saved_batch():
    batch_path = "/home/dzubke/awni_speech/speech/saved_batch/current-batch_2020-06-17.pickle"
    return read_pickle(batch_path)
       

if __name__ == "__main__":
    naren_loss()
