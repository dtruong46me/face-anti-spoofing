import argparse

import os, sys

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from src.pipelines.training_pipeline import training_pipeline


def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shape", type=tuple, default=(3,224,224))
    parser.add_argument("--num_classes", type=int, default=2)
    
    parser.add_argument("--wandb_token", type=str, default="")
    parser.add_argument("--wandb_runname", type=str, default="model1")

    parser.add_argument("--modelname", type=str, default="seresnext50")

    parser.add_argument("--train_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_training")
    parser.add_argument("--test_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_evaluation")
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()

    print("=========================================")
    print('\n'.join(f' + {k}={v}' for k, v in vars(args).items()))
    print("=========================================")

    training_pipeline(args)

if __name__=="__main__":
    run_training()