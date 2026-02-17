"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import multiprocessing as mp
import os
import subprocess as sp
import sys
import torch
from shutil import copyfile
import utils.instree_utils as instree_utils
import glob

MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_ID_CLIP = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_data_dir", type=str, help="Path to directory with the training samples")
    parser.add_argument("--node", type=str, help="which node to split (v0, v1..) the corresponding images should be under 'parent_data_dir/vi'")
    parser.add_argument("--test_name", type=str, default="test", help="your GPU id")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="your GPU id")
    parser.add_argument("--GPU_ID", type=int, default=0, help="your GPU id")
    parser.add_argument("--seed", type=int, default=0, help="your GPU id")
    parser.add_argument("--multiprocess", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="object object", help="your GPU id")
    parser.add_argument("--run_validation", action="store_true", help="your GPU id")

    # Random drop
    parser.add_argument("--random_drop", type=float, default=0.8)
    parser.add_argument("--random_drop_start_step", type=int, default=500)

    # Attention module
    parser.add_argument("--apply_otsu", action="store_true")
    parser.add_argument("--attention_start_step", type=int, default=100)
    parser.add_argument("--attention_save_step", type=int, default=50)
    parser.add_argument('--fused_res', type=int, nargs='+', default=[16])
    parser.add_argument("--ema_beta", type=float, default=0.95)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    training_data_dir = f"input_concepts/{args.parent_data_dir}/{args.node}"
    if not os.path.exists(training_data_dir):
        raise AssertionError("There is no data in " + training_data_dir)
    files = glob.glob(f"{training_data_dir}/*.png") + glob.glob(f"{training_data_dir}/*.jpg") + glob.glob(f"{training_data_dir}/*.jpeg")

    if not len(files) > 1:
        if not os.path.exists(f"{training_data_dir}/embeds.bin"):
            raise AssertionError("There is no child code in [" + training_data_dir + "/embeds.bin] to generate the data. Please run with parent node first.")
        print("Generating dataset...")
        instree_utils.generate_training_data(f"{training_data_dir}/embeds.bin", args.node, training_data_dir, device, MODEL_ID, MODEL_ID_CLIP)

    # Textual inversion
    print(f"Running with seed [{args.seed}]...")
    cmd = ["accelerate", "launch", "--gpu_ids", f"{args.GPU_ID}", "textual_inversion_decomposed.py",
                        "--train_data_dir", f"input_concepts/{args.parent_data_dir}/{args.node}",
                        "--placeholder_token", "<*> <&>",
                        "--output_dir", f"outputs/{args.parent_data_dir}/{args.node}/{args.test_name}_seed{args.seed}/",
                        "--seed", f"{args.seed}",
                        "--max_train_steps", f"{args.max_train_steps}",
                        "--initializer_token", f"{args.prompt}",
                        "--random_drop", f"{args.random_drop}",
                        "--random_drop_start_step", f"{args.random_drop_start_step}",
                        "--attention_start_step", f"{args.attention_start_step}",
                        "--attention_save_step", f"{args.attention_save_step}",
                        "--fused_res", *map(str, args.fused_res),
                        "--ema_beta", f"{args.ema_beta}",
    ]
    if args.apply_otsu:
        cmd.append("--apply_otsu")
    if args.run_validation:
        cmd += [
            "--validation_prompt", "<*>,<&>,<*> <&>",
            "--validation_steps", "100"
        ]
    exit_code = sp.run(cmd)
    
    # Saves some samples of the final node    
    instree_utils.remove_ckpts(f"outputs/{args.parent_data_dir}/{args.node}/{args.test_name}_seed{args.seed}")
    # instree_utils.save_children_nodes(args.node, f"outputs/{args.parent_data_dir}/{args.node}/{args.test_name}_seed{args.seed}/embeds/learned_embeds-steps-1000.bin", f"input_concepts/{args.parent_data_dir}", device, MODEL_ID, MODEL_ID_CLIP)
    instree_utils.save_rev_samples(f"outputs/{args.parent_data_dir}/{args.node}/{args.test_name}_seed{args.seed}", f"outputs/{args.parent_data_dir}/{args.node}/{args.test_name}_seed{args.seed}/embeds/learned_embeds-steps-1000.bin", MODEL_ID, device)