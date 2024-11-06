from ast import parse
import jittor as jt
from tqdm import tqdm
import argparse
import numpy as np
import os
from jnerf.runner import Runner, NeuSRunner, InstantNeuSRunner
from jnerf.utils.config import init_cfg, get_cfg
# jt.flags.gopt_disable=1
jt.flags.use_cuda = 1

# os.environ["disable_lock"]=1

def main():
    assert jt.flags.cuda_archs[0] >= 61, "Failed: Sm arch version is too low! Sm arch version must not be lower than sm_61!"
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val,test",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
    )

    parser.add_argument(
        "--type",
        default="novel_view",
        type=str,
    )
    parser.add_argument(
        "--mcube_threshold",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--is_continue",
        action="store_true",
        default=False,
    )
    
    args = parser.parse_args()

    assert args.type in ["novel_view","mesh","instant"],f"{args.type} not support, please choose [novel_view, mesh]"
    assert args.task in ["train","test","render", "validate_mesh"],f"{args.task} not support, please choose [train, test, render, validate_mesh]"
    is_continue = False
    if args.task == 'validate_mesh':
        is_continue = True
    else: 
        is_continue = args.is_continue

    assert args.config_file != "", "Please specify a config file"
    init_cfg(args.config_file)

    if args.type == 'novel_view':
        runner = Runner()
    elif args.type == 'mesh':
        runner = NeuSRunner(is_continue=is_continue)
    elif args.type == 'instant':
        runner = InstantNeuSRunner(is_continue=is_continue, cfg_path=args.config_file)
    else:
        print('Not support yet!')

    if args.task == "train":
        runner.train()
    elif args.task == "test":
        runner.test(True)
    elif args.task == "render":
        runner.render(True, args.save_dir)
    elif args.task == 'validate_mesh':
        runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)

if __name__ == "__main__":
    main()
