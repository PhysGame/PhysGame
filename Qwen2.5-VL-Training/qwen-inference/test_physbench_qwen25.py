from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import torch



model_name = "Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)


processor = AutoProcessor.from_pretrained(model_name)

with open("./PhysGame_880_annotation.json", "r") as f:
    anno = json.load(f)

# ========================================
#             Gradio Setting
# ========================================
result_list = []
system_prompt = "Watch the video carefully and analyze the events and object movements, focusing on any inconsistencies with physical laws. Identify and highlight instances where the behavior deviates from expected real-world physics, and select the most accurate option to describe the detected glitch.\n"
for ai in anno:
    file_name = ai['question_id']
    ground_truth = ai['answer']
    video_path = f'path/to/physbench/videos/{file_name}.mp4'
    question = ai['question']
    options = ai['options']
    prompt = system_prompt + question
    for k, v in options.items():
        prompt = prompt + '\n' + k + ": " + v
    prompt = prompt + "\nOnly give the best option."
            
    # prompt = meta_textbook + prompt # for meta textbook data
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 480 * 720,
                "fps": 2.0,
            },
            {"type": "text", "text": prompt},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    ai.update({"output": output_text[0]})
    result_list.append(ai)
    print(file_name + ': ' + output_text[0])

output_name = model_name.split("/")[-1]
with open(f"。/qwen25_output/phys_{output_name}.json", "w") as f:
    json.dump(result_list, f)
