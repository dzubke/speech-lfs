# standard libraries
import csv 
# third-party libraries
import torch
# project libraries



def visual_eval(
        model_1_path:str,
        model_2_path:str,
        model_3_path:str,
        dataset_path:str, 
        batch_size:int=8, 
        out_file:str=None
    ):
    """
    This function takes in three different models and writes their predictions along with other information,
    like the target, guess, and their respective phoneme transcriptions to a formatted txt file. 

    Arguments:
        tag - str: if best,  the "best_model" is used. if not, "model" is used. 
        add_filename - bool: if true, the filename is added to the output json
    Return:
        None
    """

    def _load_model(model_path:str)->Tuple[torch.nn.Module, dict]:
        """
        This function will load the model, config, and preprocessing object
        Args:
            model_path (str): direct path to model
        Returns:
            torch.nn.Module: torch model

        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path, preproc_path, config_path = get_names(model_path, tag=tag, get_config=True)
        
        # load and update preproc
        preproc = read_pickle(preproc_path)
        preproc.update()

        # load and assign config
        config = load_config(config_path)
        model_cfg = config['model']

        # create model
        model = CTC_train(
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
    # load all of the models

    model_dict = {
        "model_0406": model_1_path,
        "model_0902": model_2_path, 
        "model_1111": model_3_path
    }
    
    model_preproc = {
        model_name: _load_model(model_path) for model_name, model_path in model_dict.items()
    }


    with open(dataset_path, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        header = next(tsv_reader)
        tsv_dataset = list(tsv_reader)
    
    for xmpl in tsv_dataset:
        



#     ldr =  loader.make_loader(dataset_json, preproc, batch_size)


#     results = eval_loop(model, ldr)
#     print(f"number of examples: {len(results)}")
#     #results_dist = [[(preproc.decode(pred[0]), preproc.decode(pred[1]), prob)] 
#     #                for example_dist in results_dist
#     #                for pred, prob in example_dist]
#     results = [(preproc.decode(label), preproc.decode(pred), conf)
#                for label, pred, conf in results]
#     #maxdecode_results = [(preproc.decode(label), preproc.decode(pred))
#     #           for label, pred in results]
#     cer = speech.compute_cer(results, verbose=True)

#     print("PER {:.3f}".format(cer))
    
#     if out_file is not None:
#         compile_save(results, dataset_json, out_file, formatted, add_filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model."
    )
    parser.add_argument(
        "model_1_path", help="Path to oldest model."
    )
    parser.add_argument(
        "model_2_path", help="Path to older model."
    )
    parser.add_argument(
        "model_3_path", help="Path to newest model."
    )
    parser.add_argument(
        "dataset_path", help="A json file with the dataset to evaluate."
        )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size during evaluation"
    )
    parser.add_argument(
        "--save-path", help="File to save predicted results."
    )
    args = parser.parse_args()
    run(
        args.model_1_path,
        args.model_2_path,
        args.model_3_path,
        args.dataset, 
        batch_size = args.batch_size, 
        out_file=args.save
    )