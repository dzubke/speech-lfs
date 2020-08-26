#!/bin/bash
# this command should be called from inside the onnx_coreml/ directory
# command structure: bash cp_convert_zip.sh <base_path_to_model> <model_name> <num_frames> 

# assigning the arguments
model_path=$1
model_name=$2
num_frames=$3


# creating the functions
copy_files(){
    cp $1/best_model.pth ./torch_models/$2_model.pth
    cp $1/best_preproc.pyc ./preproc/$2_preproc.pyc
    cp $1/*.yaml ./config/$2_config.yaml
}

convert_model(){
    #sed -i 's/import functions\.ctc/#import functions\.ctc/g' ../speech/models/ctc_model_train.py
    python torch_to_onnx.py --model-name $1 --num-frames $2 --use-state-dict --half-precision 
    python onnx_to_coreml.py $1
    python validation.py $1 --num-frames $2
    #sed -i 's/#import functions\.ctc/import functions\.ctc/g' ../speech/models/ctc_model_train.py
}

zip_files(){
    echo "Zipping $1.zip"
    zip -j ./zips/$1.zip ./coreml_models/$1_model.mlmodel ./preproc/$1_preproc.json ./config/$1_config.yaml ./output/$1_output.json
}


# function  execution
copy_files $model_path $model_name

convert_model $model_name $num_frames

zip_files $model_name

