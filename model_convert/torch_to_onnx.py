# standard libraries
import argparse
import json
import os
# third-party libraries
import onnx
import torch
# project libraries
from get_paths import pytorch_onnx_paths
from get_test_input import generate_test_input
from import_export import torch_load, torch_onnx_export
import speech.loader as loader
from speech.models.ctc_model import CTC as CTC_model
from speech.utils.io import load_config



def torch_to_onnx(model_name:str, num_frames:int, use_state_dict:bool, return_models:bool=False):
    """
    Arguments
    -----------
    model_name: str
        filename of the model
    num_frames: int
        number of feature frames that will fix the model's size
    use_state_dict: bool
        if true, a new model will be created and the state_dict from the model in `torch_path` will loaded
    return_models: bool, False
        if true, the function will return both the torch and onnx model objects
    """  

    print(f'\nuse_state_dict: {use_state_dict}')

    freq_dim = 257  #freq dimension out of log_spectrogram 
    time_dim = num_frames

    torch_path, config_path, onnx_path = pytorch_onnx_paths(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_state_dict:
        
        config = load_config(config_path)
        model_cfg = config['model']
        
        torch_model = CTC_model(freq_dim, 39, model_cfg) 

        # TODO, dustin, replace this. only temporary before model_state_dict is saved
        state_dict_path = torch_path.replace("model.pth", "state_dict.pth")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=device)
            print(f'loaded state_dict from: {state_dict_path}')
        else:
            state_dict_model = torch.load(torch_path, map_location=device)  
            state_dict = state_dict_model.state_dict()
            print(f'loaded state_dict from: {torch_path}')

        torch_model.load_state_dict(state_dict)
        torch_model.to(device)
        print("model on cuda: ", torch_model.is_cuda)    
    else: 
        print(f'loaded entire model from: {torch_path}')
        torch_model = torch.load(torch_path, map_location=torch.device(torch_device))
        torch.save(torch_model, torch_path)
        torch_model = torch.load(torch_path, map_location=torch.device(torch_device))

    
    torch_model.eval()    
    
    input_tensor = generate_test_input("pytorch", model_name, time_dim) 
    torch_onnx_export(torch_model, input_tensor, onnx_path)
    print(f"Torch model sucessfully converted to Onnx at {onnx_path}")

    if return_models:
        onnx_model = onnx.load(onnx_path)
        return torch_model, onnx_model


if __name__ == "__main__":
    # commmand format: python pytorch_to_onnx.py <model_name> --num_frames X --use_state_dict <True/False>
    parser = argparse.ArgumentParser(description="converts models in pytorch to onnx.")
    parser.add_argument("--model-name", help="name of the model.")
    parser.add_argument("--num-frames", type=int, help="number of input frames in time dimension hard-coded in onnx model")
    parser.add_argument("--use-state-dict", action='store_true', default=False, 
                        help="boolean whether to load model from state dict") 
    args = parser.parse_args()

    return_models = False
    torch_to_onnx(args.model_name, args.num_frames, args.use_state_dict, return_models)
