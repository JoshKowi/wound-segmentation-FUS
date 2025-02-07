import argparse
from predict import predict
from evaluate_with_post_processing import exec_evaluate, evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model with post_processing")
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--predict", type=bool, default=False)
    parser.add_argument("--evaluate", type=bool, default=False)
    # parser.add_argument("--model", type=str, help="Choose which model should be used for training")
    parser.add_argument("--base_path", type=str, help="Base_path to dataset")
    parser.add_argument("--load_model", type=str, help="Filename in training_history")
    parser.add_argument("--pred_name", type=str, help="Name for predictions-directory")
    parser.add_argument("--predictions", type=str, help="Define path to predictions")
    parser.add_argument("--labels", type=str, help="Define path to groundtruth-labels")
    parser.add_argument("--input_dim", type=int, default=512)
    parser.add_argument("--thresholds", type=int, nargs='+', default=[120], help="List the thresholds to be evaluated")
    args = parser.parse_args()

    if args.train:
        pass
    if args.predict and args.evaluate:
        if args.base_path and args.pred_name and args.load_model and args.labels:
            predict(args.base_path, args.pred_name, args.load_model, input_dim=(args.input_dim, args.input_dim))
            if args.predictions:
                print(f"WARNING: ignoring path to predicitions, using predictions from model {args.load_model}")
            exec_evaluate(args.base_path + 'test/predictions/' + args.pred_name + '/', args.labels, args.thresholds)
        else:
            print("arguments missing.")
    elif args.predict:
        if args.base_path and args.pred_name and args.load_model:
            predict(args.base_path, args.pred_name, args.load_model, input_dim=(args.input_dim, args.input_dim))
        else:
            print("Arguments for prediction missing. Needs base_path, pred_name and load_model")
            exit()
    elif args.evaluate:
        if args.predictions and args.labels:
            exec_evaluate(args.predictions, args.labels, args.thresholds)
        else:
            print("Arguments for evaluation missing. Needs predictions and labels.")
            exit()