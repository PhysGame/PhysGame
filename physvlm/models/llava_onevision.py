from transformers import CLIPVisionModel, LlavaNextForConditionalGeneration
from transformers.models.llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast, \
    LlavaNextPreTrainedModel, LLAVA_NEXT_INPUTS_DOCSTRING, LLAVA_NEXT_START_DOCSTRING, LlavaNextConfig, \
    LlavaNextMultiModalProjector
from transformers.models.llava_onevision.modeling_llava_onevision import LlavaOnevisionCausalLMOutputWithPast, \
    LlavaOnevisionPreTrainedModel, LLAVA_ONEVISION_START_DOCSTRING, LLAVA_ONEVISION_INPUTS_DOCSTRING, LlavaOnevisionConfig, \
    LlavaOnevisionMultiModalProjector, get_anyres_image_grid_shape, unpad_image
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.cache_utils import Cache
_CONFIG_FOR_DOC = "LlavaOnevisionVidConfig"

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from physvlm.common.registry import registry
from physvlm.models.base_model import BaseModel

import os
import re
import gc
from PIL import Image
from physvlm.test.video_utils import load_video
from physvlm.models.llava_projector import PllavaMultiModalProjector, LLaVACLIPMultiModalProjector, \
    LLaVACLIPCLSMultiModalProjector, LLaVACLIPClsNceMultiModalProjector, LLaVACLIPNceMultiModalProjector, \
    PllavaSelf3DMultiModalProjector, LLaVACLIP3DMultiModalProjector, LLaVACLIP3DOptMultiModalProjector, \
    LLaVACLIP3DNCEMultiModalProjector, LLaVACLIPMulti3DMultiModalProjector, LLaVASIGLIP3DMultiModalProjector, \
    LlavaOnevisionMultiModalProjector
import einops
from physvlm.models.siglip_btadapter import SiglipVisionModel_BTAdapter
from physvlm.models.siglip import SiglipModelwithVideo
from safetensors import safe_open

from peft import (
    LoraConfig,
    get_peft_model,
)
import math
import numpy as np
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.image_processing_utils import select_best_resolution
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel



class LlavaOnevisionVidConfig(LlavaOnevisionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.video_input = "mean"
        self.pooling = 'avp'
        self.pllava_pooling_shape=None
        self.pooling_kernel=None
        self.pooling_stride=None
        self.frame_shape=(24,24)
        self.clip_weight = None
        self.clip_post_pretrain = None
        self.btadapter = False
        self.btadapter_depth = 4
        self.max_T = 128
        self.long_clip = False
        self.pooling_temp = 0.01
        self.pad_token_id = 151643

@add_start_docstrings(
    """The LLaVA-Onevision model which consists of a vision backbone and a language model.""",
    LLAVA_ONEVISION_START_DOCSTRING,
)

@registry.register_model("llava_vid_qwen_nogrid")
class LlavaNextVidForConditionalGenerationQwenNoGrid(LlavaNextPreTrainedModel, BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {}
    def __init__(self, config: LlavaOnevisionVidConfig):
        super().__init__(config)
        if config.btadapter:
            config.vision_config.depth = config.btadapter_depth
            config.vision_config.max_T = config.max_T
            self.vision_tower = SiglipVisionModel_BTAdapter._from_config(config.vision_config)
        else:
            self.vision_tower = AutoModel.from_config(
                config.vision_config, attn_implementation=config._attn_implementation
            )

        self.pooling = config.pooling 
        if config.pooling == 'avp':
            self.multi_modal_projector = LlavaOnevisionMultiModalProjector(config)
        elif config.pooling == 'pllava':
            self.multi_modal_projector = PllavaMultiModalProjector(config)
        elif config.pooling == 'clipST' or config.pooling == 'clipT':
            self.multi_modal_projector = LLaVACLIPMultiModalProjector(config)
        elif config.pooling == 'clipcls':
            self.multi_modal_projector = LLaVACLIPCLSMultiModalProjector(config)
        elif config.pooling == 'clipcls_nce':
            self.multi_modal_projector = LLaVACLIPClsNceMultiModalProjector(config)
        elif config.pooling == 'clipST_nce' or config.pooling == 'clipT_nce':
            self.multi_modal_projector = LLaVACLIPNceMultiModalProjector(config)
        elif config.pooling == 'pllava_self':
            self.multi_modal_projector = PllavaSelf3DMultiModalProjector(config)
        elif config.pooling == 'clipST_3d' or config.pooling == 'clipST_3Dbefore':
            self.multi_modal_projector = LLaVACLIP3DMultiModalProjector(config)
        elif config.pooling == 'clipST_3d_opt':
            self.multi_modal_projector = LLaVACLIP3DOptMultiModalProjector(config)
        elif config.pooling == 'clipST_3d_nce':
            self.multi_modal_projector = LLaVACLIP3DNCEMultiModalProjector(config)  
        elif config.pooling == 'clipST_3d_multi':
            self.multi_modal_projector = LLaVACLIPMulti3DMultiModalProjector(config)  

        embed_std = 1 / math.sqrt(config.text_config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.text_config.hidden_size, dtype=self.dtype) * embed_std)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.video_input = config.video_input

        #self.num_frames = config.num_frames

        self.config.hidden_size = config.text_config.hidden_size
        self.post_init()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder
    def get_decoder(self):
        return self.language_model.get_decoder()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.tie_weights
    def tie_weights(self):
        return self.language_model.tie_weights()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.resize_token_embeddings
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration._merge_input_ids_with_image_features
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)
        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
     
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=f'cuda:{inputs_embeds.device}' if isinstance(inputs_embeds.device, int) else inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=f'cuda:{inputs_embeds.device}' if isinstance(inputs_embeds.device, int) else inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=f'cuda:{input_ids.device}' if isinstance(input_ids.device, int) else input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device),
            non_image_indices.to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device),
            text_to_overwrite.to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device),
        )
        attention_mask = attention_mask.to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device)
       
        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device)


        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def pack_image_features(self, image_features, image_sizes, image_newline=None, vision_aspect_ratio="anyres_max_9"):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
            vision_aspect_ratio (`str`, *optional*, "anyres_max_9"):
                Aspect ratio used when processong image features. The default value is "anyres_max_9".
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                max_num_patches = int(vision_aspect_ratio.strip("anyres_max_"))
                channels, curr_height, curr_width = image_feature.shape
                ratio = math.sqrt(curr_height * curr_width / (max_num_patches * height**2))
                if ratio > 1.1:
                    image_feature = image_feature[None]
                    image_feature = nn.functional.interpolate(
                        image_feature, [int(curr_height // ratio), int(curr_width // ratio)], mode="bilinear"
                    )[0]
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.stack(new_image_features, dim=0)
        # image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens
    
    # Copied from transformers.models.llava_next.modeling_llava_next.image_size_to_num_patches
    def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
        """
        Calculate the number of patches after the preprocessing for images of any resolution.

        Args:
            image_size (`torch.LongTensor` or `np.ndarray` or `Tuple[int, int]`):
                The size of the input image in the format (height, width). ?
            grid_pinpoints (`List`):
                A list containing possible resolutions. Each item in the list should be a tuple or list
                of the form `(height, width)`.
            patch_size (`int`):
                The size of each image patch.

        Returns:
            int: the number of patches
        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints should be a list of tuples or lists")

        # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
        if not isinstance(image_size, (list, tuple)):
            if not isinstance(image_size, (torch.Tensor, np.ndarray)):
                raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
            image_size = image_size.tolist()

        best_resolution = select_best_resolution(image_size, grid_pinpoints)
        height, width = best_resolution
        num_patches = 0
        # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                num_patches += 1
        # add the base patch
        num_patches += 1
        return num_patches

    @add_start_docstrings_to_model_forward(LLAVA_NEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=LlavaNextCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        clip_ids: Optional[torch.Tensor] = None,
        clip_mask: Optional[torch.Tensor] = None,
        clip_answer_ids: Optional[torch.Tensor] = None,
        clip_answer_mask: Optional[torch.Tensor] = None,
        dpo_forward: Optional[bool] = False,
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        vision_aspect_ratio = (self.config.vision_aspect_ratio)

        if labels is not None:
            labels = labels[:,:input_ids.size(1)]
        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                batch_size, num_frames, num_patches, num_channels, height, width = pixel_values.shape

                if self.config.btadapter:
                    reshaped_pixel_values = pixel_values.view(batch_size*num_patches, num_frames , num_channels, height, width)
                else:
                    reshaped_pixel_values = pixel_values.view(batch_size * num_frames * num_patches, num_channels, height, width)
                
                #reshaped_pixel_values = pixel_values.view(batch_size * num_frames * num_patches, num_channels, height, width)

                image_features = self.vision_tower(reshaped_pixel_values, output_hidden_states=True)
                selected_image_feature = image_features.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature

                nce_loss = None
                if self.pooling == 'avp':
                    image_features = self.multi_modal_projector(selected_image_feature)  #torch.Size([80, 576, 1024])
                    image_features = einops.rearrange(image_features, '(B T) L D -> B T L D', T=num_frames)
                    if self.video_input=="mean":
                        image_features = image_features.mean(1)
                    #Qwen ([4, 729, 3584])
                    #torch.Size([6, 576, 4096])
                elif self.pooling == 'pllava':
                    image_features = self.multi_modal_projector(selected_image_feature,
                                                            batch_size=batch_size, num_frames=num_frames)

                elif self.pooling == 'clipST' or self.pooling == 'clipT':
                    image_features = self.multi_modal_projector(selected_image_feature,
                                                            clip_ids, clip_mask, num_frames, self.pooling)
                elif self.pooling == 'clipcls':
                    image_features = self.multi_modal_projector(selected_image_feature, image_features[1],
                                                            clip_ids, clip_mask, num_frames)
                elif self.pooling == 'clipcls_nce':
                    image_features, nce_loss = self.multi_modal_projector(selected_image_feature, image_features[1],
                                                            clip_ids, clip_mask, clip_answer_ids, clip_answer_mask, num_frames)

                elif self.pooling == 'clipST_nce' or self.pooling == 'clipT_nce':
                    image_features, nce_loss = self.multi_modal_projector(selected_image_feature, clip_ids, clip_mask,
                                                                           clip_answer_ids, clip_answer_mask, num_frames, self.pooling)
                elif self.pooling == 'pllava_self':
                    image_features = self.multi_modal_projector(selected_image_feature,
                                                            batch_size=batch_size, num_frames=num_frames)

                elif self.pooling == 'clipST_3d' or self.pooling == 'clipST_3Dbefore':
                    after = (self.pooling == 'clipST_3d')
                    image_features = self.multi_modal_projector(selected_image_feature,
                                                            clip_ids, clip_mask, num_frames, after)
                elif self.pooling == 'clipST_3d_opt':    
                    image_features = self.multi_modal_projector(selected_image_feature, clip_ids, clip_mask,
                                                                           clip_answer_ids, clip_answer_mask, num_frames)
                elif self.pooling == 'clipST_3d_nce':
                    image_features, nce_loss = self.multi_modal_projector(selected_image_feature, clip_ids, clip_mask,
                                                                           clip_answer_ids, clip_answer_mask, num_frames)
                elif self.pooling == 'clipST_3d_multi':
                    image_features = self.multi_modal_projector(selected_image_feature,
                                                            clip_ids, clip_mask, num_frames)
                
                # print(image_features.shape) #[2, 1024, 4096]

                # image_features = torch.split(image_features, image_num_patches, dim=0)
                # image_num_patches = [
                #     image_size_to_num_patches(
                #         image_size=imsize,
                #         grid_pinpoints=self.config.image_grid_pinpoints,
                #         patch_size=self.config.vision_config.image_size,
                #     )
                #     for imsize in image_sizes
                # ]
                # image_num_patches = [1] * batch_size
                # image_features = torch.split(image_features, image_num_patches, dim=0)
                # image_features, feature_lens = self.pack_image_features(
                # image_features,
                # image_sizes,
                # image_newline=self.image_newline,
                # vision_aspect_ratio=vision_aspect_ratio,
                # )

                # special_image_mask = (
                #     (input_ids == self.config.image_token_index)
                #     .unsqueeze(-1)
                #     .expand_as(inputs_embeds)
                #     .to(inputs_embeds.device)
                # )
                # image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                # inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(f'cuda:{torch.long}' if isinstance(torch.long, int) else torch.long)

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=f'cuda:{attention_mask.device}' if isinstance(attention_mask.device, int) else attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        # print(self.language_model)
        # print(inputs_embeds.shape)
        # assert 1 <= 0
        # import pdb
        # pdb.set_trace()
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]
        if dpo_forward:
            return logits, labels
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(f'cuda:{logits.device}' if isinstance(logits.device, int) else logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(f'cuda:{labels.device}' if isinstance(labels.device, int) else labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(f'cuda:{shift_logits.device}' if isinstance(shift_logits.device, int) else shift_logits.device)
            )

            if nce_loss is not None:
                loss += 0.001 * nce_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        clip_ids = kwargs.get("clip_ids", None)
        clip_mask = kwargs.get("clip_mask", None)
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
                "clip_ids": clip_ids,
                "clip_mask": clip_mask,
            }
        )
        return model_inputs
        
    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration._reorder_cache
    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)

    def load_pretrained_weight(self, ckpt_path):
        if os.path.isdir(ckpt_path):
            ckpt = self.get_state_dict(ckpt_path)
            msg = self.load_state_dict(ckpt, strict=False)
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = self.load_state_dict(ckpt, strict=False)

    @classmethod
    def get_state_dict(self, path, prefix='(model|non_lora_trainable)'):
        pattern = re.compile(r'^(model|non_lora_trainable).*?(\.safetensors|\.bin)$')
        matching_files = [filename for filename in os.listdir(path) if pattern.match(filename)]

        model_state_dict = {}
        for model_path in matching_files:
            if model_path.endswith('safetensors'):
                with safe_open(os.path.join(path,model_path), framework="pt", device='cpu') as f:
                    for k in f.keys():
                        model_state_dict[k] = f.get_tensor(k)
            elif model_path.endswith('bin'):
                partial_state_dict = torch.load(os.path.join(path,model_path), map_location=torch.device('cpu'))
                model_state_dict.update(partial_state_dict)
        return model_state_dict

    @classmethod
    def from_config(cls, cfg):
        llama_model = cfg.get("llama_model")
        pretrained_state_dict = cls.get_state_dict(llama_model)
        pretrained_cfg = LlavaOnevisionVidConfig.from_pretrained(llama_model)

        pretrained_cfg.video_input = cfg.get("video_input","mean")   
        pretrained_cfg.pooling = cfg.get("pooling","avp") 
        pretrained_cfg.clip_weight = cfg.get("clip_weight", None)
        pretrained_cfg.clip_post_pretrain = cfg.get("clip_post_pretrain", None)
        pretrained_cfg.btadapter = cfg.get("btadapter",False)
        pretrained_cfg.btadapter_depth = cfg.get("btadapter_depth",4)
        pretrained_cfg.max_T = cfg.get("max_T",64)
        pretrained_cfg.long_clip = cfg.get("long_clip",False)
        pretrained_cfg.pooling_temp = cfg.get("pooling_temp",0.01)
        pretrained_cfg.pad_token_id = cfg.get("pad_token_id",-1)
        

        if cfg.get("pllava_pooling_shape",None):
            pretrained_cfg.pllava_pooling_shape = eval(cfg.get("pllava_pooling_shape"))
        if cfg.get("pooling_kernel",None):
            pretrained_cfg.pooling_kernel = eval(cfg.get("pooling_kernel"))
        if cfg.get("pooling_stride",None):
            pretrained_cfg.pooling_stride = eval(cfg.get("pooling_stride"))
        if cfg.get("frame_shape",None):
            pretrained_cfg.frame_shape = eval(cfg.get("frame_shape"))
        
        model = cls(pretrained_cfg).to(f'cuda:{torch.float16}' if isinstance(torch.float16, int) else torch.float16)
        msg = model.load_state_dict(pretrained_state_dict, strict=False)
        print(msg)
        del pretrained_state_dict
        gc.collect()

        if cfg.get("btadapter",False):
            model.vision_tower.vision_model.init_weights()

        if cfg.get("use_lora", False):
            kwargs = {}
            
            kwargs.update({"target_modules": ["q_proj", "v_proj"]})
            peft_config = LoraConfig(
                task_type="CAUSAL_LM", inference_mode=False, 
                r=cfg.get("lora_r"), lora_alpha=cfg.get("lora_alpha"), lora_dropout=cfg.get("lora_dropout"),
                **kwargs
            )
            model.language_model = get_peft_model(model.language_model, peft_config)
            model.language_model.print_trainable_parameters()
            
        if cfg.get("freeze_LLM",True):
            for n,p in model.language_model.named_parameters():
                if 'lora' not in n:
                    p.requires_grad = False
        
        if cfg.get("freeze_vision_tower",True):
            for n,p in model.vision_tower.named_parameters():
                if 'btadapter' not in n:
                    p.requires_grad = False   
        if 'clip' in cfg.get("pooling","avp"):
            if cfg.get("freeze_vproj",True):
                for p in model.multi_modal_projector.visual_projection.parameters():
                    p.requires_grad = False   
            if cfg.get("freeze_tproj",True):
                for p in model.multi_modal_projector.text_projection.parameters():
                    p.requires_grad = False  
            if cfg.get("freeze_text",True):
                for p in model.multi_modal_projector.clip_text_model.parameters():
                    p.requires_grad = False       
        
        if cfg.get("gradient_checkpointing",False):
            model.gradient_checkpointing_enable()
        
        bt_adapter_weight = cfg.get("bt_adapter_weight", "")  
        if bt_adapter_weight:
            clipmodel = CLIPModelwithVideo.from_pretrained(bt_adapter_weight)
            btadapter_model = clipmodel.vision_model
            model.vision_tower.vision_model.load_state_dict(btadapter_model.state_dict())

        if cfg.get("onlyLLM", False):
            for n,p in model.named_parameters():
                if 'language_model' not in n:
                    p.requires_grad = False
        
        if cfg.get("LLMandProj", False):
            for n,p in model.named_parameters():
                if 'language_model' not in n and 'multi_modal_projector' not in n:
                    p.requires_grad = False
        
        if cfg.get("LLMandBTA", False):
            for n,p in model.named_parameters():
                if 'language_model' not in n and 'btadapter' not in n:
                    p.requires_grad = False

        ckpt_path = cfg.get("ckpt", "") 
        if ckpt_path:
            #if os.path.isdir(ckpt_path) and not cfg.get("use_lora", False):
            #    model = model.from_pretrained(ckpt_path)
            #elif cfg.get("use_lora", False):
            if os.path.isdir(ckpt_path):
                ckpt = cls.get_state_dict(ckpt_path)
                msg = model.load_state_dict(ckpt, strict=False)
            else:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                msg = model.load_state_dict(ckpt, strict=False)
        #[n for n,p in model.named_parameters() if p.requires_grad==True]
        return model
        # model.to("cuda:0")
        # # prepare image and text prompt, using the appropriate prompt template
        # from transformers import LlavaOnevisionProcessor
        # from PIL import Image
        # import requests
        # # model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype="float16", device_map="cuda:0")
        # processor = LlavaOnevisionProcessor.from_pretrained(llama_model)
        # prompt = "USER: <image>\nWhat is shown in this video? ASSISTANT:"
        # # raw_frames = load_video('/home/thr/code/S3/PLLaVA/example/working.mp4', num_frm=cfg.get("num_frames",1))
        # image_file = "/home/thr/code/S3/PLLaVA/example/bear.jpg"
        # raw_image = Image.open(image_file)
        # inputs = processor(text=prompt, images=None, return_tensors='pt').to(0, torch.float16)

        # output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        # print(processor.batch_decode(output, skip_special_tokens=True)[0])
        # # inputs = processor(prompt, raw_frames, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        # output = model.generate(**inputs, max_new_tokens=100)

        # print(processor.decode(output[0], skip_special_tokens=True))
        