
import logging
from tqdm import tqdm
import os
from physvlm.common.dist_utils import get_rank, get_world_size, init_distributed_mode
import argparse
import numpy as np

import torch
import torch.distributed as dist
import transformers

from physvlm.common.config import Config
from physvlm.test.video_utils import LLaVA_Processer, STLLM_Processer
from physvlm.test.PhysGame_bench.utils import PGbench_dataset, infer_pgbench_llava, check_ans, save_results
from physvlm.common.registry import registry

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--ckpt-path", help="path to checkpoint file.", default="")
    parser.add_argument("--num-frames", type=int, required=False, default=100)
    parser.add_argument("--specified_item", type=str, required=False, default=None)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument('--data_anno', type=str, help='Directory to benchmark JSON annotation.', required=True)
    parser.add_argument('--video_dir', type=str, help='Directory to benchmark videos.', required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--all_token", action='store_true')
    parser.add_argument("--system_llm", action='store_false')
    parser.add_argument("--ask_simple", action='store_true')
    parser.add_argument("--llava", action='store_true')
    return parser.parse_args()

def load_model_and_dataset(rank, world_size, args):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    #model_config.ckpt = args.ckpt_path
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    model.load_pretrained_weight(args.ckpt_path)
    model = model.to('cuda:{}'.format(rank))
    for name, para in model.named_parameters():
        para.requires_grad = False
    model.eval()

    dataset = PGbench_dataset(data_anno=args.data_anno, video_dir=args.video_dir, num_segments=args.num_frames)
    dataset.set_rank_and_world_size(rank, world_size)

    if args.llava:
        processor = LLaVA_Processer(model_config)
    else:
        processor = STLLM_Processer(model_config)

    return model, processor, dataset

def run(rank, args, world_size):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, processor, dataset = load_model_and_dataset(rank,
                                                       world_size,
                                                       args)

    if rank == 0:
        tbar = tqdm(total=len(dataset))

    correct = 0
    total = 0
    result_list = []
    done_count = 0

    infer_pgbench = infer_pgbench_llava
    for example in dataset:
        total += 1
        pred = infer_pgbench(
            model, processor,
            example, 
            system="Watch the video carefully and analyze the events and object movements, focusing on any inconsistencies with physical laws. Identify and highlight instances where the behavior deviates from expected real-world physics, and select the most accurate option to describe the detected glitch.\n",
            question_prompt="\nOnly give the best option.",
            answer_prompt="Best option:(",
            return_prompt='(',
            system_q=False,
            print_res=False,
            system_llm=True
        )
        gt = example['answer']
        video_id = example['video_id']
        class_anno = example['class']
        subclass_anno = example['subclass']
        result_list.append({'video_id': video_id, 'pred': pred, 'gt': gt, 'question': example['question'], 'class_anno': class_anno, 'subclass_anno':subclass_anno})
            
        if check_ans(pred=pred, gt=gt):
            correct += 1
        if rank == 0:
            tbar.update(len(result_list) - done_count)
            tbar.set_description_str(
                f" Chunk Total Acc: {correct / total * 100 :.2f}%"
            )
            done_count = len(result_list)
    return result_list

def main():
    args = parse_args()

    args.distributed = True
    args.dist_url = "env://"
    init_distributed_mode(args)
    rank, world_size = get_rank(), get_world_size()
    if not os.path.exists(args.output_dir) and rank==0:
        os.makedirs(args.output_dir)

    local_result = run(rank, args, world_size)
    gather_list = [None for _ in range(world_size)]
    # Gather results at all ranks
    dist.all_gather_object(gather_list, local_result)
    result_list = []
    for res in gather_list:
        result_list.extend(res)
    if rank == 0:
        save_results(result_list, args.output_dir, args.output_name)
    
if __name__ == "__main__":
    main()