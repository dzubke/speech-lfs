# standard libraries
import argparse
import os
# third-party libraries
from coremltools.models.neural_network import quantization_utils
import onnx
from onnx_coreml import convert


def main(model_name:str, half_precision:bool, quarter_precision:bool):

    onnx_path = os.path.join("onnx_models", model_name+"_model.onnx")
    coreml_path = os.path.join("coreml_models", model_name+"_model.mlmodel")

    onnx_model = onnx.load(onnx_path)

    coreml_model = convert(model=onnx_model,
                            minimum_ios_deployment_target = '13')

    if half_precision:
        coreml_model = quantization_utils.quantize_weights(coreml_model, nbits=16) 
    if quarter_precision:
        coreml_model = quantization_utils.quantize_weights(coreml_model, nbits=8)

    coreml_model.save(coreml_path)
    print(f"Onnx model successfully converted to CoreML at: {coreml_path}")


if __name__ == "__main__":
    # command format  python onnx_to_coreml.py <model_name>
    parser = argparse.ArgumentParser(description="converts onnx model to coreml.")
    parser.add_argument("model_name", help="name of the model.")
    parser.add_argument("--half-precision", action='store_true', default=False,  help="converts the model to half precision.")
    parser.add_argument("--quarter-precision", action='store_true', default=False, help="converts the model to quarter precision.")
    args = parser.parse_args()

    main(args.model_name, args.half_precision, args.quarter_precision)
