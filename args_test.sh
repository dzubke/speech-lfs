#!/bin/bash

POSITIONAL=()
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
    -n|--name)
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
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo "FILE EXTENSION  = ${MODEL_PATH}"
echo "SEARCH PATH     = ${MODEL_NAME}"
echo "LIBRARY PATH    = ${NUM_FRAMES}"
echo "DEFAULT         = ${BEST_TAG}"
echo "Positional      = ${POSITIONAL}"

