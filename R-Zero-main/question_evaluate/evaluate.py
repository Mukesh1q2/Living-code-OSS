import json
import vllm
from transformers import AutoTokenizer
import argparse
import re
from evaluation.datasets_loader import get_dataset_handler
from mathruler.grader import extract_boxed_content, grade_answer
import os
# math_verify = get_dataset_handler("math")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Base")
parser.add_argument("--num_samples", type=int, default=10)
parser.add_argument("--suffix", type=str, default="77")
parser.add_argument("--save_name", type=str, default="")
args = parser.parse_args()
STORAGE_PATH = os.getenv("STORAGE_PATH")
print('start load')
with open(f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json", "r") as f:
    data = json.load(f)
os.remove(f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json")
# data  = [    {
#         "question": "Solve the equation \\(|x^2 - 3x + 2| = |x - 1.5|\\). How many real solutions does this equation have?",
#         "answer": "4",
#         "score": 0
#     }]
def extract_answer(response):
    return re.search(r"\\boxed{(.*?)}", response).group(1)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = vllm.LLM(
    model=args.model,
    tokenizer=args.model,
    gpu_memory_utilization=0.85,
    seed=int(args.suffix),
)
sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.num_samples,
)
wrong_data = [item for item in data if item['score'] == -1]
correct_data = [item for item in data if item['score'] == 0]  
questions = [item["question"] for item in correct_data]
answers = [item["answer"] for item in correct_data]
chats = [[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": question}] for question in questions]
if tokenizer.chat_template:
    prompts = [tokenizer.apply_chat_template(chat, tokenize=False,add_generation_prompt=True, add_special_tokens=True) for chat in chats]
else:
    prompts = ["system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"] for chat in chats]
responses = model.generate(prompts, sampling_params=sample_params,use_tqdm=True)
print(len(data))
results_all = []
for response, answer, question in zip(responses, answers, questions):

    try:
        error_flag = False
        count = 0
        results = [extract_boxed_content(output.text) for output in response.outputs]
        answer_counts = {}
        for result in results:
            found_match = False
            # Check if result matches any existing answer group
            try:
                for existing_answer in answer_counts:
                    if grade_answer(result, existing_answer) or grade_answer(existing_answer, result) or result == existing_answer or ('no ' in result.lower() and 'no ' in existing_answer.lower()):
                        answer_counts[existing_answer] += 1
                        found_match = True
                        break
            except:
                error_flag = True
                break
            # If no match found, create new answer group
            if not found_match:
                answer_counts[result] = 1
        # print(answer_counts)
        # Find the answer with the most matches
        if error_flag:
            continue
        if answer_counts:
            max_count = max(answer_counts.values())
            majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        # print(majority_answer)
        score = max_count/len(results)
        # print(score)
        # print(majority_answer)
        if "证明" in question or 'box' in question.lower() or 'text' in majority_answer.lower():
            continue
        results_all.append({"question": question, "answer": majority_answer, "score": score, 'results':results})
    except Exception as e:
        print("Error:", e)
        continue
    # print({"question": question, "answer": majority_answer, "score": score, 'results':results})
    # print(score,question,flush=True)
print(len(results_all))

with open(f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}_results.json", "w") as f:
    json.dump(results_all, f, indent=4)

    # break
