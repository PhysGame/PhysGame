import argparse
import torch
from mmengine.fileio import FileClient
client = FileClient('disk')
from decord import VideoReader, cpu
import io
from PIL import Image
import numpy as np

# imports modules for registration
from physvlm.datasets.builders import *
from physvlm.models import *
from physvlm.processors import *
from physvlm.runners import *
from physvlm.tasks import *

from physvlm.common.config import Config
from physvlm.conversation.conv import conv_templates
from physvlm.test.video_utils import LLaVA_Processer
from physvlm.common.registry import registry
from physvlm.test.vcgbench.utils import VideoChatGPTBenchDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.", default="config/physvlm_dpo_training.yaml")
    parser.add_argument("--gpu-id", type=int, default=7, help="specify the gpu to load the model.")
    parser.add_argument("--ckpt-path", help="path to ckpt file.", default='./ckpt/checkpoint_dpo')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def read_video(video_path, bound=None, num_segments=32):
    video_bytes = client.get(video_path)
    vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    images_group = list()
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments) 
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)

    return images_group

# ========================================
#             Model Initialization
# ========================================
print('Initializing Chat')

args = parse_args()
cfg = Config(args)
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config)
model.load_pretrained_weight(args.ckpt_path)
model = model.to('cuda:{}'.format(args.gpu_id))
model.to(torch.bfloat16)
for name, para in model.named_parameters():
    para.requires_grad = False
model.eval()

processor = LLaVA_Processer(model_config)
processor.processor.image_processor.set_reader(video_reader_type='decord', num_frames=32)
conv = conv_templates['conv_vcg_qwen']
print('Initialization Finished')

# ========================================
#             Answer Generation
# ========================================
num_frames = 32
video_path = './asset/xbmgbr.mp4'
prompt = "What violates physical commonsense in this video?"
img_list = read_video(video_path, num_segments=num_frames)
chat_state = conv.copy()
chat_state.user_query(prompt, is_mm=True)

full_question = chat_state.get_prompt()
inputs = processor(full_question, prompt, img_list)

inputs = inputs.to(model.device)
if conv.sep[0]=='<|im_end|>\n': #qwen
    split_str = 'assistant\n'
    target_dtype = model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
    inputs['pixel_values'] = inputs.pixel_values.to(f'cuda:{target_dtype}' if isinstance(target_dtype, int) else target_dtype)
else:
    split_str = conv.roles[1]
output = model.generate(**inputs, num_beams=5, temperature=1.0, max_new_tokens=200)
llm_message = processor.processor.decode(output[0], skip_special_tokens=True)
llm_message = llm_message.split(split_str)[1].strip()

print(f'PhysVLM answer: {llm_message}')
