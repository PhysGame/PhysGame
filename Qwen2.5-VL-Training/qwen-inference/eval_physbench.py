import json
import argparse


parser = argparse.ArgumentParser(description="This is a sample command line parser for reading result.")


parser.add_argument('--file_path', type=str, help="output inference json file name")  # 可选标志
args = parser.parse_args()

with open('classes_final_v2.json', 'r', encoding='utf-8') as f:
    categories_data = json.load(f)

with open(args.file_path, 'r', encoding='utf-8') as f:
    scores_data = json.load(f)


for score_di in scores_data:
    if score_di["output"][0] == score_di["answer"]:
        score_di.update({"score":1})
    else:
        score_di.update({"score":0})

scores_dict = {item['question_id']: item['score'] for item in scores_data}

def calculate_average_scores_and_accuracy(category_data):
    averages = {}
    total_correct = 0
    total_items = 0

    for main_category, subcategories in category_data.items():
        for subcategory, question_dict in subcategories.items():

            scores = []

            for sub_label, question_ids in question_dict.items():

                scores.extend([scores_dict.get(qid, 0) for qid in question_ids])
                
                total_correct += sum(1 for qid in question_ids if scores_dict.get(qid, 0) == 1)
                total_items += len(question_ids)
            
            if scores:
                average_score = sum(scores) / len(scores)
            else:
                average_score = 0
            
            averages[subcategory] = average_score
            #print(f"{subcategory}:{len(scores)}-----{sum(scores)}")
    
    overall_accuracy = total_correct / total_items if total_items else 0
    return averages, overall_accuracy

averages, overall_accuracy = calculate_average_scores_and_accuracy(categories_data)

print(f"\nOverall:{overall_accuracy * 100:.6f}%")

print("Average:")
for subcategory, avg_score in averages.items():
    print(f"{subcategory}: {avg_score* 100:.6f}%")

print("--------------------------------------------------------------------------------------------")
accuracy_per_subcategory, overall_accuracy = calculate_average_scores_and_accuracy(categories_data)
output_data = [round(overall_accuracy * 100, 1)]
for subcategory in accuracy_per_subcategory.values():
    output_data.append(round(subcategory * 100, 1))

output_str = ' & '.join(map(str, output_data))
print(output_str)