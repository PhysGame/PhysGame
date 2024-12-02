import os
import json
import torch

from physvlm.test.video_utils import EvalDataset

from physvlm.conversation.mvbench_conversation import ask, answer, EasyDict, get_prompt2


class PGbench_dataset(EvalDataset):
    def __init__(self, data_anno, video_dir, num_segments=8, resolution=224):
        super().__init__(num_segments=num_segments)
        self.data_list = []
        with open(data_anno, 'r') as f:
            self.data_list = json.load(f)
        self.num_segments = num_segments
        self.resolution = resolution
        self.video_dir = video_dir
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
    

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for ch, c in data['options'].items():
            question += f"({ch}) {c}\n"
        question = question.rstrip()
        answer = f"({answer}) {data['options'][answer]}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method['video']
        bound = None
        video_path = os.path.join(self.video_dir, self.data_list[idx]['question_id']) + ".mp4"
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx])
            
        return {
            'video_id': self.data_list[idx]['question_id'],
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'class': self.data_list[idx]['class_anno'],
            'subclass': self.data_list[idx]['subclass_anno']
        }

def infer_pgbench_llava(
    model, processor,
    data_sample, 
    system="", 
    question_prompt='', # add in the end of question
    answer_prompt=None, # add in the begining of answer
    return_prompt='',  # add in the begining of return message
    system_q=False, # whether add question in the system prompt for QFormer
    print_res=True,
    system_llm=False
    ):

    video = data_sample["video"]

    role = ("USER", "ASSISTANT")
    chat = EasyDict({
        "system": system,
        "roles": role,
        "messages": [],
        "sep": "###"
    })

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>\n"])
    
    if system_llm:
        prompt = system + data_sample['question'] + question_prompt
    else:
        prompt = data_sample['question'] + question_prompt
    
    
    ask(prompt, chat)

    chat.messages.append([chat.roles[1], answer_prompt])
    prompt = get_prompt2(chat).split('</Video>\n')[1]
    prompt = "###USER:"+ " <image>\n"+ prompt.split('###USER:')[1]

    inputs = processor(prompt, data_sample['question'], video)
    inputs = inputs.to(model.device).to(torch.float16)
    output = model.generate(**inputs, max_new_tokens=200)
    llm_message = processor.processor.decode(output[0], skip_special_tokens=True).split(answer_prompt)[1]
    #llm_message = llm_message.split(role[1])[1]

    # remove potential explanation
    llm_message = return_prompt + llm_message.strip()
    return llm_message


def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

def save_results(result_list, save_path, save_name):
    correct, total = 0, 0
    output_list = []
    subclass_cnt = {}
    for res in result_list:
        total += 1
        pred = res['pred']
        gt = res['gt']
        subclass = res['subclass_anno']
        if check_ans(pred=pred, gt=gt):
            correct += 1
            output_list.append({'video_id': res['video_id'], 'pred': pred, 'gt': gt, 'question': res['question'], 'class_anno': res['class_anno'], 'subclass_anno': subclass, "score":1})
            if subclass not in subclass_cnt.keys():
                subclass_cnt.update({subclass: [1, 1]})
            else:
                subclass_cnt[subclass][0] += 1
                subclass_cnt[subclass][1] += 1
        else:
            output_list.append({'video_id': res['video_id'], 'pred': pred, 'gt': gt, 'question': res['question'], 'class_anno': res['class_anno'], 'subclass_anno': subclass, "score":0})
            if subclass not in subclass_cnt.keys():
                subclass_cnt.update({subclass: [0, 1]})
            else:
                subclass_cnt[subclass][1] += 1
    all_results = {
        "acc": correct / total * 100,
        "output": output_list
    }
    
    print (f'Total Acc: {correct / total * 100 :.2f}%', )
    for sub_i in subclass_cnt.keys():
        print(f'{sub_i} Acc: {subclass_cnt[sub_i][0] / subclass_cnt[sub_i][1] * 100 :.2f}%')
        
    with open(os.path.join(save_path, f"{save_name}.json"), 'w') as f:
        json.dump(all_results, f)