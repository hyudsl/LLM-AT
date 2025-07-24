import json
import argparse
import sys
sys.path.append('/home/injae/LLM-AT')
from utils.model_util import load_model, cost_evaluator
from utils.MCQA_util import MCQA_evaluator
from Module import GJ_Module


def main(llm_type, dataset, iter_flag, max_iter, huggingface_model, task_type):

    tokenizer, model = None, None
    if huggingface_model:
        tokenizer, model = load_model(huggingface_model, llm_type) 

    baseline = GJ_Module(llm_type, model, tokenizer, huggingface_model)

    final_result = []
    while iter_flag <= max_iter:
        if iter_flag < max_iter:
            print(f'=============== inference {iter_flag} start ===============')
            dataset, iter_flag = baseline.Generator(dataset, iter_flag, task_type)
            dataset, finish_dataset, iter_flag = baseline.Judge(dataset, iter_flag, task_type)
 
            final_result += finish_dataset

            if len(dataset) == 0:
                print(f'=============== inference {iter_flag} end ===============')
                break
            iter_flag +=1

        elif iter_flag == max_iter: 
            print(f'=============== iter {iter_flag} start ===============')
            dataset, iter_flag = baseline.Generator(dataset, iter_flag, task_type)

            for item in dataset:
                item['validity'] = '-'
                judgement_history = item.get('judgement result', [])
                judgement_history.append("-")
                item['judgement result'] = judgement_history
                
            final_result += dataset
            
            print(f'=============== inference {iter_flag} end ===============')
            iter_flag +=1
    
    return final_result



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Generator and Judge for dataset")
    parser.add_argument("--model_name",
                        type=str,
                        default="gpt-4o-mini",
                        help="Name of the model to use")
    
    parser.add_argument("--base_path",
                        type=str,
                        default="/home/injae/LLM-AT/result/MCQA/",
                        help="Base path for save the project")

    parser.add_argument("--dataset_name",
                        type=str,
                        default='MCQA',
                        help="Dataset")
    
    parser.add_argument("--iter_flag",
                        type=int,
                        default=1,
                        help="Iteration flag")
    
    parser.add_argument("--max_iter",
                        type=int,
                        default=1,
                        help="Max iteration")
    
    parser.add_argument("--huggingface_model",
                        type=bool,
                        default=False,
                        help="Use Huggingface model")

    parser.add_argument("--save",
                        type=bool,
                        default=True,
                        help="Whether to save the final result")
    
    args = parser.parse_args()

    llm_type = args.model_name
    dataset_name = args.dataset_name
    base_path = args.base_path
    iter_flag = args.iter_flag
    max_iter = args.max_iter
    huggingface_model = args.huggingface_model
    save = args.save

    dataset = json.load(open(f'/home/injae/LLM-AT/dataset/{dataset_name}/sample/{dataset_name}.json')) 
   
    for n, item in enumerate(dataset, start=1):
        item['uid'] = n 

    final_result = main(llm_type, dataset, iter_flag, max_iter, huggingface_model, dataset_name)
    text, final_acc, wrong = MCQA_evaluator(final_result)
    total_cost, total_time = cost_evaluator(final_result)

    if save:
        if max_iter == 1:
            baseline_type='Single'
        else: 
            baseline_type=f'Iteration_{max_iter}'

        save_path = base_path+ f'{dataset_name}_{baseline_type}_{llm_type}_POT_.json'
        with open(save_path, "w") as saver:
            json.dump(final_result, saver, indent=4)


    print(f'====== Baseline: {dataset_name} ======')
    print(f'Accuracy: {final_acc} % ({text})')
    print(f'Total Cost: $ {total_cost}')
    print(f'Total Execution Time: {total_time} minutes')