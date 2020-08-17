# standard libraries
import argparse
import json
import os
# third-party libraries
import torch
# project libraries
from get_paths import pytorch_onnx_paths
from get_test_input import generate_test_input
from import_export import torch_load, torch_onnx_export
import speech.loader as loader
from speech.models.ctc_model import CTC as CTC_model
from speech.utils.convert import convert_half_precision
from speech.utils.io import load_config



def main(model_name, num_frames, use_state_dict, half_precision):  
    print(f'\nuse_state_dict: {use_state_dict}')
    print(f'\nuse half precision: {half_precision}')

    freq_dim = 257  #freq dimension out of log_spectrogram 
    time_dim = num_frames

    torch_path, config_path, onnx_path = pytorch_onnx_paths(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_state_dict:
        print(f'loaded state_dict from: {torch_path}')
        
        config = load_config(config_path)
        model_cfg = config['model']
        
        ctc_model = CTC_model(freq_dim, 39, model_cfg) 
        state_dict_model = torch.load(torch_path, map_location=device)
        ctc_model.load_state_dict(state_dict_model.state_dict())
    
    else: 
        print(f'loaded entire model from: {torch_path}')
        ctc_model = torch.load(torch_path, map_location=torch.device(torch_device))
        torch.save(ctc_model, torch_path)
        ctc_model = torch.load(torch_path, map_location=torch.device(torch_device))
    
    # converts model to half precision
    if half_precision:
       ctc_model = convert_half_precision(ctc_model)
        
    
    ctc_model.eval()    
    
    input_tensor = generate_test_input("pytorch", model_name, time_dim, half_precision) 
    torch_onnx_export(ctc_model, input_tensor, onnx_path)
    print(f"Torch model sucessfully converted to Onnx at {onnx_path}")



if __name__ == "__main__":
    # commmand format: python pytorch_to_onnx.py <model_name> --num_frames X --use_state_dict <True/False>
    parser = argparse.ArgumentParser(description="converts models in pytorch to onnx.")
    parser.add_argument("model_name", help="name of the model.")
    parser.add_argument("--num_frames", type=int, help="number of input frames in time dimension hard-coded in onnx model")
    parser.add_argument("--use_state_dict", action='store_true', default=False, 
                        help="boolean whether to load model from state dict") 
    parser.add_argument("--half-precision", action='store_true', default=False, 
                        help="boolean whether to convert model to half precision") 
    args = parser.parse_args()

    main(args.model_name, args.num_frames, args.use_state_dict, args.half_precision)
