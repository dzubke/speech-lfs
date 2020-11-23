# standard libraries
import argparse
from collections import OrderedDict
import os
import csv 
from typing import Tuple
# third-party libraries
import torch
import tqdm
# project libraries
import speech.loader
from speech.models.ctc_decoder import decode as ctc_decode
from speech.models.ctc_model_train import CTC_train as CTC_model
from speech.utils.data_helpers import lexicon_to_dict, text_to_phonemes
from speech.utils.io import get_names, load_config, load_state_dict, read_pickle



def visual_eval(config:dict)->None:
    """
    This function takes in three different models and writes their predictions along with other information,
    like the target, guess, and their respective phoneme transcriptions to a formatted txt file. 
    Config contains:
        models: contains model_1 through model_3 with name, path, tag, and model_name  for the `get_names` function
        dataset_path (str): path to evaluation dataset
        save_path (str): path where the formatted txt file will be saved
        lexicon_path (str): path to lexicon
        n_top_beams (int): number of beams output from the ctc_decoder
    Return:
        None
    """

    # unpack the config
    model_params = config['models']
    dataset_path = config['dataset_path']
    output_path = config['output_path']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the models and preproc objects
    print(f"model_params contains: {model_params}")

    model_preproc = {
        model_name: _load_model(params, device) for model_name, params in model_params.items()
    }

    output_dict = {}     # dictionary containing the printed outputs

    # open the tsv file that contains the data on each example
    with open(dataset_path, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        header = next(tsv_reader)
        tsv_dataset = list(tsv_reader)
        # tsv header is: "id", "target", "guess", "lessonId", "lineId", "uid", "redWords_score", "date"
        for xmpl in tsv_dataset:
            output_dict.update({
                xmpl[0]:{               # record id
                    "target": xmpl[1],
                    "guess": xmpl[2]
                }
            })

    output_dict = add_phonemes(output_dict, config['lexicon_path'])

    # directory where audio paths are stored
    audio_dir = os.path.join(os.path.dirname(dataset_path), "audio")

    # loop through each output file and perform inference for the 3 models
    for rec_id in tqdm.tqdm(output_dict.keys()):
        audio_path = os.path.join(audio_dir, rec_id + ".wav")
        dummy_target = []   # dummy target list fed into the preprocessor, not used
        output_dict[rec_id]['infer'] = {}   # initialize a dict for the inference outputs

        for model_name, (model, preproc) in model_preproc.items():
            with torch.no_grad():    # no gradients calculated to speed up inference
                inputs, dummy_target = preproc.preprocess(audio_path, dummy_target)
                inputs = torch.FloatTensor(inputs)
                inputs = torch.unsqueeze(inputs, axis=0).to(device)   # add the batch dim and push to `device`
                probs, _ = model(inputs, softmax=True)      # don't need rnn_args output in `_`
                probs = probs.data.cpu().numpy().squeeze() # convert to numpy and remove batch-dim
                top_beams = ctc_decode(probs, 
                                        beam_size=3, 
                                        blank=model.blank, 
                                        n_top_beams=config['n_top_beams']
                )
                top_beams = [(preproc.decode(preds), probs) for preds, probs in top_beams]
                output_dict[rec_id]['infer'].update({model_name: top_beams})


    # sort the dictionary to ease of matching audio file with formatted output
    output_dict = OrderedDict(sorted(output_dict.items()))
    with open(output_path, 'w') as out_file:  
        for rec_id in output_dict.keys():  
            out_file.write(f"rec_id:\t\t\t{rec_id}\n")
            out_file.write(f"target:\t\t\t{output_dict[rec_id]['target']}\n")
            out_file.write(f"guess:\t\t\t{output_dict[rec_id]['guess']}\n")
            out_file.write(f"tar_phones:\t\t{output_dict[rec_id]['tar_phones']}\n")
            out_file.write(f"ges_phones:\t\t{output_dict[rec_id]['ges_phones']}\n")
            # loop through the models and the top beams for each model
            for model_name in output_dict[rec_id]['infer'].keys():
                top_beam=True
                for preds, confid in output_dict[rec_id]['infer'][model_name]:
                    if top_beam:
                        out_file.write(f"{model_name}:\t({round(confid, 2)})\t{preds}\n")
                        top_beam = False
                    else:
                        out_file.write(f"\t   \t({round(confid, 2)})\t{preds}\n")
            #out_file.write(f"2020-11-18:\t\t {output_dict[rec_id]['model_1118']}\n")
            #out_file.write(f"2020-09-25:\t\t {output_dict[rec_id]['model_0925']}\n")
            #out_file.write(f"2020-09-02:\t\t {output_dict[rec_id]['model_0902']}\n")
            #out_file.write(f"2020-04-06:\t\t {output_dict[rec_id]['model_0406']}\n")
            out_file.write("\n\n")

def add_phonemes(output_dict:dict, lexicon_path:str):
    """
    This function takes in the output_dict and updates the dict to include the phoneme labels 
    of the target and guess sentences
    Args:
        output_dict (dict): keys are `record_id` and values are `target` and `guess` sentences
    Returns:
        dict: updated `output_dict` with phoneme labels for `target` and `guess`
    """
    lexicon = lexicon_to_dict(lexicon_path)
    for xmpl in output_dict.keys():
        output_dict[xmpl]['tar_phones'] = text_to_phonemes(output_dict[xmpl]['target'], 
                                                            lexicon,
                                                            unk_token="<UNK>"
        )
        output_dict[xmpl]['ges_phones'] = text_to_phonemes(output_dict[xmpl]['guess'], 
                                                            lexicon,
                                                            unk_token="<UNK>"
        )
    return output_dict



def _load_model(model_params:str, device)->Tuple[torch.nn.Module, speech.loader.Preprocessor]:
    """
    This function will load the model, config, and preprocessing object and prepare the model and preproc for evaluation
    Args:
        model_path (dict): dict containing model path, tag, and filename
        device (torch.device): torch processing device
    Returns:
        torch.nn.Module: torch model
        preprocessing object (speech.loader.Preprocessor): preprocessing object
    """

    model_path, preproc_path, config_path = get_names(
        model_params['path'], 
        tag=model_params['tag'], 
        get_config=True,
        model_name=model_params['filename']
    )
    
    # load and update preproc
    preproc = read_pickle(preproc_path)
    preproc.update()

    # load and assign config
    config = load_config(config_path)
    model_cfg = config['model']
    model_cfg.update({'blank_idx': config['preproc']['blank_idx']}) # creat `blank_idx` in model_cfg section

    # create model
    model = CTC_model(
        preproc.input_dim,
        preproc.vocab_size,
        model_cfg
    )

    state_dict = load_state_dict(model_path, device=device)
    model.load_state_dict(state_dict)

    model.to(device)
    # turn model and preproc to eval_mode
    model.set_eval()
    preproc.set_eval()

    return model, preproc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model."
    )
    parser.add_argument(
        "--config", help="Path to config file containing the necessary inputs"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    visual_eval(config)
