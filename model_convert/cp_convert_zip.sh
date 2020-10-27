#!/bin/bash
# this command should be called from inside the model_convert/ directory
# command structure: bash cp_convert_zip.sh <base_path_to_model> <model_name> <num_frames> 

# assigning the arguments

display_help(){
    echo "$(basename "$0") [-mp MODEL_PATH] [-mn MODEL_NAME] [-nf NUM_FRAMES] 
            (options) [--quarter-precision] [--half-precision] [--best]" >& 2
    echo "call this function inside the <main>/model_convert directory"
    echo "where:
            -mp or --model-path:  is the path to the model directory
            -mn or --model-name: is the model name
            -nf or --num-frames: is the number of frames
            --quarter-precision: for quarter precision
            --half-precision:    for half precision
            --best: to use the best_model in directory
    "
}


while [[ $# -gt 0 ]]
do
    key="$1"
    VALUE="$2"

    case $key in
        -mp|--model-path)
        MODEL_PATH="$VALUE"
        shift # past argument
        shift # past value
        ;;
        -mn|--model-name)
        MODEL_NAME="$VALUE"
        shift # past argument
        shift # past value
        ;;
        -nf|--num-frames)
        NUM_FRAMES="$VALUE"
        shift # past argument
        shift # past value
        ;;
        --best)
        BEST_TAG="best_"
        shift # past argument
        ;;
        --half-precision)
        HALF_PRECISION="--half-precision"
        shift # past argument
        ;;
        --quarter-precision)
        QUARTER_PRECISION="--quarter-precision"
        shift # past argument
        ;;
        --help|-h)
        display_help
        exit 1
        ;;
        -*)
        echo "Error: Unknown option: $1" >&2
        display_help
        exit 1
        ;;
    esac
done


# creating the functions

copy_files(){

    MODEL_PATH=$1
    MODEL_NAME=$2
    BEST_TAG=$3

    cp ${MODEL_PATH}/${BEST_TAG}model_state_dict.pth ./torch_models/${MODEL_NAME}_model.pth
    echo "copied ${MODEL_PATH}/${BEST_TAG}model_state_dict.pth to ./torch_models/${MODEL_NAME}_model.pth"
    cp ${MODEL_PATH}/${BEST_TAG}preproc.pyc ./preproc/${MODEL_NAME}_preproc.pyc
    cp ${MODEL_PATH}/*.yaml ./config/${MODEL_NAME}_config.yaml
}

convert_model(){
    
    MODEL_NAME=$1
    NUM_FRAMES=$2
    HALF_PRECISION=$3
    QUARTER_PRECISION=$4

    sed -i '' 's/import functions\.ctc/#import functions\.ctc/g' ../speech/models/ctc_model_train.py
    python torch_to_onnx.py --model-name "$MODEL_NAME" --num-frames $NUM_FRAMES --use-state-dict 
    python onnx_to_coreml.py $MODEL_NAME $HALF_PRECISION $QUARTER_PRECISION
    python validation.py $MODEL_NAME --num-frames $NUM_FRAMES
    sed -i '' 's/#import functions\.ctc/import functions\.ctc/g' ../speech/models/ctc_model_train.py
}

zip_files(){
    # $1 IS MODEL_NAME

    echo "Zipping $1.zip"
    zip -j ./zips/$1.zip ./coreml_models/$1_model.mlmodel ./preproc/$1_preproc.json ./config/$1_config.yaml ./output/$1_output.json
}

cleanup(){
    # is run if SIGINT is sent
    # ensure the modified ctc_model_train file is put back to original state
    sed -i '' 's/#import functions\.ctc/import functions\.ctc/g' ../speech/models/ctc_model_train.py
}


# function  execution

# runs the cleanup finction if SIGINT is entered
trap cleanup SIGINT

copy_files $MODEL_PATH $MODEL_NAME $BEST_TAG

convert_model $MODEL_NAME $NUM_FRAMES $HALF_PRECISION $QUARTER_PRECISION

zip_files $MODEL_NAME

