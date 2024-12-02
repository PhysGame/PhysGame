from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.models.llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast, \
    LlavaNextPreTrainedModel, LLAVA_NEXT_INPUTS_DOCSTRING, LLAVA_NEXT_START_DOCSTRING, LlavaNextConfig, \
    LlavaNextMultiModalProjector, get_anyres_image_grid_shape, unpad_image
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.cache_utils import Cache
_CONFIG_FOR_DOC = "LlavaNextVidConfig"

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from physvlm.common.registry import registry
from physvlm.models.base_model import BaseModel

import os
import re
from PIL import Image
from physvlm.test.video_utils import load_video
from physvlm.models.llava_projector import PllavaMultiModalProjector
import einops

@add_start_docstrings(
    """The LLAVA-NeXT model which consists of a vision backbone and a language model.""",
    LLAVA_NEXT_START_DOCSTRING,
)

class LlavaNextVidConfig(LlavaNextConfig):
    def __init__(self, vision_config=None, text_config=None, ignore_index=-100, image_token_index=32000, projector_hidden_act="gelu", vision_feature_select_strategy="default", vision_feature_layer=-2, image_grid_pinpoints=None, **kwargs):
        super().__init__(vision_config, text_config, ignore_index, image_token_index, projector_hidden_act, vision_feature_select_strategy, vision_feature_layer, image_grid_pinpoints, **kwargs)
        self.video_input = "mean"
        self.pooling = 'avp'
        self.pllava_pooling_shape=(16,8,8)
        self.frame_shape=(24,24)


@registry.register_model("llava_vid")
class LlavaNextVidForConditionalGeneration(LlavaNextPreTrainedModel, BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {}
    def __init__(self, config: LlavaNextVidConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.pooling = config.pooling 
        if config.pooling == 'avp':
            self.multi_modal_projector = LlavaNextMultiModalProjector(config)
        elif config.pooling == 'pllava':
            self.multi_modal_projector = PllavaMultiModalProjector(config)

        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size, dtype=self.dtype))

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
        if labels is not None:
            labels = labels[:,:input_ids.size(1)]

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                batch_size, num_frames, num_patches, num_channels, height, width = pixel_values.shape
                reshaped_pixel_values = pixel_values.view(batch_size * num_frames * num_patches, num_channels, height, width)
                image_features = self.vision_tower(reshaped_pixel_values, output_hidden_states=True)

                selected_image_feature = image_features.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature

                if self.pooling == 'avp':
                    image_features = self.multi_modal_projector(selected_image_feature)  #torch.Size([80, 576, 1024])
                    image_features = einops.rearrange(image_features, '(B T P) L D -> (B P) T L D', T=num_frames, P=num_patches)
                    if self.video_input=="mean":
                        image_features = image_features.mean(1)
                    height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                    num_frames = 1
                elif self.pooling == 'pllava':
                    image_features = self.multi_modal_projector(selected_image_feature,
                                                            batch_size=batch_size * num_patches, num_frames=num_frames)
                    height = width = self.multi_modal_projector.pooling_shape[-1]
                    num_frames = self.multi_modal_projector.pooling_shape[0]
                # split up image_features for each of the individual images
                # hence we get a list of image_features, each of shape (5, num_patches, hidden_size)
                # if we assume each image has 5 image features (base image + 4 patches)
                split_sizes = [image.shape[1] for image in pixel_values]
                image_features = torch.split(image_features, split_sizes, dim=0)

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
                

                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]

                        if height * width * num_frames != base_image_feature.shape[0]:
                            raise ValueError("The number of patches is not consistent with the image size.")
                        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                            image_sizes[image_idx],
                            self.config.image_grid_pinpoints,
                            self.config.vision_config.image_size,
                        )
                        image_feature = einops.rearrange(image_feature, '(PH PW) (T H W) D -> PH PW T H W D', T=num_frames, PH=num_patch_height, H=height)
                        #image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1) #torch.Size([2, 2, 24, 24, 4096])
                        image_feature = einops.rearrange(image_feature, 'PH PW T H W D -> (T D) PH H PW W')
                        #image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx]) #480 854
                        image_feature = einops.rearrange(image_feature, '(T D) H W -> D T H W', T=num_frames)
                        image_feature = torch.cat(            #4096, 28, 48
                            (
                                image_feature,
                                self.image_newline[:, None, None, None].expand(*image_feature.shape[:-1], 1),
                            ),
                            dim=-1,
                        ) #torch.Size([4096, 28, 49])
                        image_feature = einops.rearrange(image_feature, 'D T H W -> (T H W) D', T=num_frames)
                        #image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)
                    new_image_features.append(image_feature)
                image_features = torch.stack(new_image_features, dim=0)
                #image_features = einops.rearrange(image_features, '(B T) L D -> B T L D', T=num_frames)
                #if self.video_input=="mean":
                #    image_features = image_features.mean(1)

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

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
            }
        )
        return model_inputs

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration._reorder_cache
    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)

    @classmethod
    def get_state_dict(self, path, prefix='model'):
        pattern = re.compile(f'{prefix}-(\d+)-of-(\d+).safetensors')
        matching_files = [filename for filename in os.listdir(path) if pattern.match(filename)]

        model_state_dict = {}
        for model_path in matching_files:
            partial_state_dict = torch.load(os.path.join(path,model_path), map_location=torch.device('cpu'))
            model_state_dict.update(partial_state_dict)
        return model_state_dict

    @classmethod
    def from_config(cls, cfg):
        llama_model = cfg.get("llama_model")
        pretrained_model = LlavaNextForConditionalGeneration.from_pretrained(llama_model, torch_dtype=torch.float16) 
        pretrained_cfg = LlavaNextVidConfig.from_pretrained(llama_model)

        pretrained_cfg.video_input = cfg.get("video_input","mean")   
        pretrained_cfg.pooling = cfg.get("pooling","avp") 
        if cfg.get("pllava_pooling_shape",None):
            pretrained_cfg.pllava_pooling_shape = eval(cfg.get("pllava_pooling_shape"))
        if cfg.get("frame_shape",None):
            pretrained_cfg.frame_shape = eval(cfg.get("frame_shape"))
        
        model = cls(pretrained_cfg).to(f'cuda:{torch.float16}' if isinstance(torch.float16, int) else torch.float16)
        model.load_state_dict(pretrained_model.state_dict(), strict=True)

        if cfg.get("freeze_LLM",True):
            for p in model.language_model.parameters():
                p.requires_grad = False
        if cfg.get("freeze_vision_tower",True):
            for p in model.vision_tower.parameters():
                p.requires_grad = False   
        
        if cfg.get("gradient_checkpointing",False):
            model.gradient_checkpointing_enable()
        
        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            if os.path.isdir(ckpt_path):
                model = model.from_pretrained(ckpt_path)
            else:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                msg = model.load_state_dict(ckpt, strict=False)
        return model
        #model.to("cuda:0")
        ## prepare image and text prompt, using the appropriate prompt template
        #processor = LlavaNextProcessor.from_pretrained(llama_model)
        #prompt = "USER: <image>\nWhat is shown in this video? ASSISTANT:"
        #raw_frames = load_video('/group/40034/ruyangliu/code_reading/PLLaVA/example/working.mp4', num_frm=cfg.get("num_frames",1))
#
        #inputs = processor(prompt, raw_frames, return_tensors="pt").to("cuda:0")
#
        ## autoregressively complete prompt
        #output = model.generate(**inputs, max_new_tokens=100)
#
        #print(processor.decode(output[0], skip_special_tokens=True))
        