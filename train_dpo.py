# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
import argparse
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Any, Tuple, Union
from accelerate.utils import DistributedType
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import transformers
from transformers import CLIPTokenizer, SiglipTokenizer
import tokenizers
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

from physvlm.train.constants import IGNORE_INDEX, X_TOKEN_INDEX, DEFAULT_X_TOKEN, X_INDEX_TOKEN, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from physvlm.train.stllm_trainer import LLaVADPOTrainer
from physvlm.datasets.data_utils import load_jsonl, load_json
import physvlm.tasks as tasks
from physvlm.common.config import Config
from physvlm.datasets.datasets.llavavid_processor import LlavaNextVidProcessor, LlavaOnevisionVidProcessor
from physvlm.conversation import conv as conversation_lib
conversation_lib.default_conversation = conversation_lib.conv_templates["plain_qwen"]
from physvlm.train.utils import DPODataCollatorWithPadding

from PIL import Image
os.environ["WANDB_DISABLED"]="true"



local_rank = None

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--local_rank", required=False, default=0)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    X: Optional[List[str]] = field(default=None)
    image_tower: Optional[str] = field(default=None)
    video_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_x_start_end: bool = field(default=False)
    mm_use_x_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    both: bool = False
    reject_both: bool = False
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    training_modal: Optional[str] = field(default='video')
    num_sample: Optional[int] = field(default=None)
    num_frames: Optional[int] = field(default=16)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    fix_vit: bool = True
    dpo_alpha: float = field(default=1.0)
    beta: float = field(default=0.1)
    gamma: float = field(default=1.0)
    generate_during_eval: bool = field(default=False)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def tokenizer_X_token(prompt, tokenizer, X_token_index, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(f'<{X_INDEX_TOKEN[X_token_index].lower()}>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [X_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['multi_modal_projector', 'btadapter', 'vision_resampler']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def save_my_lora_ckpt(output_dir, args, model):
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(), args.lora_bias
    )
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    if args.local_rank == 0 or args.local_rank == -1:
        model.config.save_pretrained(output_dir)
        model.save_pretrained(output_dir, state_dict=state_dict)
        torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def make_conv(prompt, answer):
    return [
        {
            "from": "human",
            "value": prompt,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]


def get_text_len(tokenizer, text):
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

def preprocess_v1_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    X: str = None
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv._append_message(role, sentence["value"])
        conversations.append(conv.get_prompt() + conv.sep[0])

    # Tokenize conversations

    if X is not None:
        input_ids = torch.stack([tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX[X], return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    # assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep[0] + conv.roles[1] + ": "
    result_targets = []
    for conversation, target in zip(conversations, targets):
        # print(conversation, target)
        sep1 = conv.roles[0] 
        sep2 = conv.roles[1] 
        raw_text = conversation.split(sep2)
        for idx in range(0, len(raw_text)-1):
            raw_text[idx] = raw_text[idx] + sep2
            
        cur_len = get_text_len(tokenizer, raw_text[0])
        target[:cur_len] = -100
        for text in raw_text[1:-1]: 
            total_len = get_text_len(tokenizer, text)
            ans_len = get_text_len(tokenizer, text.split(sep1)[0].rstrip(' '))
            target[(cur_len+ans_len):(cur_len+total_len)] = -100
            cur_len += total_len
        cur_len += get_text_len(tokenizer, raw_text[-1].rstrip(' '))
        assert cur_len == target.shape[0], f"The final length ({cur_len}) is not equal to the original prompt ({target.shape[o]}): {conversation}"
        target = F.pad(target, (0, 3000 - len(target)), value=-100)
        result_targets.append(target)
        


    return dict(
        input_ids=input_ids,
        labels=result_targets,
    )

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if has_image and "<image>" in sentence["value"]:
                assert sentence["value"].startswith("<image>"), print(sentence["value"])

                _input_id = tokenizer(role).input_ids + nl_tokens + [IMAGE_TOKEN_INDEX] + nl_tokens + tokenizer(sentence["value"][len("<image>") :]).input_ids + [im_end] + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == "<|im_start|>assistant":
                _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        # input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        # target += [IGNORE_INDEX] * (max_len - len(target))
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
        # attention_mask=input_ids.ne(tokenizer.pad_token_id), # tensor(bs x seq_len)
    )

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    X: str
) -> Dict:
    DEFAULT_TOKEN = DEFAULT_X_TOKEN[X]
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX[X], return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_X_token(source[0]['value'], tokenizer, X_TOKEN_INDEX[X]))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_X: str = None
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # print(sources)
    # return preprocess_qwen(sources, tokenizer, has_image=False)
    X = has_X if has_X is None else has_X.upper()
    # if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
    #     return preprocess_plain(sources, tokenizer, X=X)
    # elif conversation_lib.default_conversation.version.startswith("v1"):
    return preprocess_v1_qwen(sources, tokenizer, X=X)
    # else:
    #     raise NotImplementedError

def load_data(data_args):
    if 'jsonl' in data_args.data_path:
        data_list = load_jsonl(data_args.data_path)
    else: 
        data_list = load_json(data_args.data_path)
    return data_list

class DPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(Dataset, self).__init__()
        list_data_dict = load_data(data_args)
        if data_args.num_sample is not None:
            list_data_dict = list_data_dict[:data_args.num_sample]

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.training_modal = data_args.training_modal

    def __len__(self):
        # return 20
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if any([x.lower() in sample for x in DEFAULT_X_TOKEN.keys()]) else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if any([x.lower() in sample for x in DEFAULT_X_TOKEN.keys()]) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        '''
        {
            'prompt': 'Is there a snowman wearing a green scarf and hat in the background?',
            'chosen': 'No, there is no snowman wearing a green scarf and hat in the background of the image. The image features a person ...',
            'rejected': 'No, there is no snowman in the background.',
            'image_path': '/mnt/bn/liangkeg/data/ruohongz/dpo_data/dpo_images/LRVInstruction-000000009569.jpg',
            'image_name': 'LRVInstruction-000000009569.jpg'
        }
        '''
        try:
            has_X = None
            data_dict = copy.deepcopy(self.list_data_dict[i])
            if self.training_modal == 'image':
                image_file = data_dict['frame']
                image_folder = self.data_args.image_folder
                processor = self.data_args.image_processor
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                prompt = data_dict['prompt']
                prompt = prompt.replace("<image>", "").strip()
                prompt = "<image>\n" + prompt
                data_dict['prompt'] = prompt
                has_X = 'image'

            elif self.training_modal == 'video':
                video_file = data_dict['video']
                video_folder = self.data_args.video_folder
                processor = self.data_args.video_processor
                video = os.path.join(video_folder, video_file)
                # print(video)
                video = processor(video, return_tensors='pt')
                prompt = data_dict['prompt']
                prompt = prompt.replace("<video>", "").strip()
                prompt = "<image>\n" + prompt
                data_dict['prompt'] = prompt
                has_X = 'video'
            else:
                raise("Training modal not supported")

            data_dict['has_X'] = has_X
            if has_X == 'image':
                data_dict['image'] = image
            elif has_X == 'video':
                data_dict['video'] = video
                # print('success video')
            
            return data_dict
        except Exception as e:
            print(f'Error with {e}, {self.list_data_dict[i]}')
            return self.__getitem__(random.randint(0, self.__len__()-1))

@dataclass
class DPODataCollator(DPODataCollatorWithPadding):
    def collate(self, batch):

        padded_batch = {}
        for k in batch[0].keys():
            if (k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels")) and 'clip' not in k :
                # if "prompt" in k:
                #     to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                # else:
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                else:
                    padded_batch[k] = torch.stack(to_pad)
                    continue

                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(to_pad, batch_first=True, padding_value=padding_value)

            else:
                padded_batch[k] = [ex[k] for ex in batch]
        for k in ['chosen_input_ids', 'rejected_input_ids']:
            attn_k = k.replace('input_ids', 'attention_mask')
            padded_batch[attn_k] = padded_batch[k].ne(self.tokenizer.pad_token_id)
            
        return padded_batch


    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        has_X: str = None,
        is_long_clip: bool = False
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        # import pdb; pdb.set_trace()
        batch = {}
        
        chosen_sources = make_conv(prompt, chosen)
        rejected_sources = make_conv(prompt, rejected)
        chosen_data_dict = preprocess(
            [chosen_sources],
            self.tokenizer,
            has_X=has_X
        )
        #chosen_data_dict['attention_mask'] = chosen_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        rejected_data_dict = preprocess(
            [rejected_sources],
            self.tokenizer,
            has_X=has_X
        )
        #rejected_data_dict['attention_mask'] = rejected_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        chosen_data_dict = {k: v[0] for k, v in chosen_data_dict.items()}
        rejected_data_dict = {k: v[0] for k, v in rejected_data_dict.items()}

        for k, toks in {
            "chosen": chosen_data_dict,
            "rejected": rejected_data_dict,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        if self.clip_tokenizer is not None:
            self.clip_length = 196 if is_long_clip else 64
            if not self.both:
                question = prompt.split('<image>\n')[1]
                clip_input = self.clip_tokenizer(question, return_tensors='pt', max_length=self.clip_length)
                batch.update(
                    {
                        'clip_input_ids': clip_input.input_ids.squeeze(),
                        # 'clip_attention_mask': clip_input.attention_mask.squeeze(),
                    }
                )
            else:
                clip_chosen = prompt.split('<image>\n')[1]+' '+chosen
                clip_rejected = prompt.split('<image>\n')[1]+' '+rejected if self.reject_both else prompt.split('<image>\n')[1] 
                clip_input_chosen = self.clip_tokenizer(clip_chosen, return_tensors='pt', max_length=self.clip_length)
                clip_input_rejected = self.clip_tokenizer(clip_rejected, return_tensors='pt', max_length=self.clip_length)
                batch.update(
                    {
                        'clip_chosen_input_ids': clip_input_chosen.input_ids.squeeze(),
                        'clip_rejected_input_ids': clip_input_rejected.input_ids.squeeze(),
                        # 'clip_chosen_attention_mask': clip_input_chosen.attention_mask.squeeze(),
                        # 'clip_rejected_attention_mask': clip_input_rejected.attention_mask.squeeze(),
                    }
                )
        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []
        Xs, keys = [], []
        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]
            has_X = feature['has_X']
            Xs.append(feature[has_X])
            keys.append(has_X)
             
            batch_element = self.tokenize_batch_element(prompt, chosen, rejected, has_X=has_X)
            tokenized_batch.append(batch_element)

        # return collated batch
        padded_batch =  self.collate(tokenized_batch)
        padded_batch['images'] = [Xs, keys]  # we do not change the key's name.
        return padded_batch


def make_dpo_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = DPODataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    return train_dataset

def merge_dict_to_argv(input_dict):
    input_dict.pop('task')
    i = 0
    while i < len(sys.argv):
        if sys.argv[i].startswith('--cfg-path'):
            sys.argv.pop(i)
            sys.argv.pop(i)
            break
        else:
            i += 1
    sys.argv.extend([f'--{key}={value}' for key, value in input_dict.items()])

def train():
    global local_rank

    cfg = Config(parse_args())
    model_cfg = cfg.model_cfg

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    merge_dict_to_argv(cfg.run_cfg)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if training_args.bits in [4, 8]:
        raise NotImplementedError("Quantization is not supported yet.")

    training_args.lora_enable = model_cfg.get('use_lora', False)
    training_args.tune_mm_mlp_adapter = (not model_cfg.get('use_lora', False)) and model_cfg.get('freeze_LLM', True)
    model_args.X = ["Image", "Video"]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_cfg.llama_model,
        cache_dir=training_args.cache_dir,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["plain_qwen"]

    
    # data_args.video_processor = LlavaNextVidProcessor.from_pretrained(model_cfg.llama_model).set_reader(video_reader_type='rawframe',num_frames=data_args.num_frames)
    data_args.video_processor = LlavaOnevisionVidProcessor.from_pretrained(model_cfg.llama_model).set_reader(video_reader_type='decord', num_frames=data_args.num_frames)
    data_args.is_multimodal = True

    train_dataset = make_dpo_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    if model_cfg.clip_weight:
        clip_tokenizer = SiglipTokenizer.from_pretrained(model_cfg.clip_weight)
        clip_tokenizer.model_max_length = 248 if model_cfg.long_clip else 77
    else:
        clip_tokenizer = None

    data_collator = DPODataCollator(
            tokenizer,
            label_pad_token_id=IGNORE_INDEX,
            pad_token_id=tokenizer.pad_token_id,
            clip_tokenizer=clip_tokenizer,
            both=data_args.both,
            reject_both=data_args.both,
        )

    trainer = LLaVADPOTrainer(
        model,
        args=training_args,
        dpo_alpha=training_args.dpo_alpha,
        beta=training_args.beta,
        gamma=training_args.gamma,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        generate_during_eval=False, #training_args.generate_during_eval,
    )
    trainer.save_my_lora_ckpt = save_my_lora_ckpt


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
