import argparse
import os, sys

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from src.pipelines.evaluation_pipeline import evaluation_pipeline

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="/kaggle/input/lcc-fasd/LCC_FASD/LCC_FASD_evaluation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_checkpoint", type=str, default="/kaggle/working/checkpoint/cvproject.ckpt")
    parser.add_argument("--input_shape", type=tuple, default=(3,224,224))
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()

    print("=========================================")
    print('\n'.join(f' + {k}={v}' for k, v in vars(args).items()))
    print("=========================================")


    evaluation_pipeline(args)

if __name__=="__main__":
    run_evaluation()