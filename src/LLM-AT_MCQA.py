from tqdm import tqdm
import json
import sys
import argparse
sys.path.append('/home/injae/LLM-AT')
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_chroma import Chroma
from src.Module import GJ_Module, S_Module
from utils.simulation_util import build_result_history, build_metadata
from utils.model_util import cost_evaluator
from utils.MCQA_util import MCQA_evaluator


def LLM_AT(dataset, tier_list: list, top_k: int, th: float, lambda_: float, acc_list: list, vectorstore, task_name):

    base_module = GJ_Module(model=None, tokenizer=None, huggingface_model=False)
    starter = S_Module(tier_list, top_k, th, lambda_, acc_list)
    final_result = []
    idx = 1

    for item in tqdm(dataset):
        result_history = {}
        api_trajectory = []
        api_flag = 0
        validity = 'no'
        
        question = item['question']

        if idx <= top_k:
            while validity != 'yes':
                api_trajectory.append(tier_list[api_flag])
                item = base_module.Generator(item, tier_list[api_flag], task_name)
                
                if tier_list[api_flag] != tier_list[-1]: 

                    if tier_list[api_flag] == tier_list[0]:
                        item, validity = base_module.Judge(item, tier_list[api_flag+1], task_name)
                        result_history[tier_list[api_flag]] = (item['execute'] if task_name == "MATH" else item['pred_idx'], validity)
                        api_flag +=1
                    else: 
                        item, validity = base_module.Judge(item, tier_list[api_flag], task_name)
                        result_history[tier_list[api_flag]] = (item['execute'] if task_name == "MATH" else item['pred_idx'], validity)
                        api_flag +=1 
                    
                else:
                    validity = 'yes'
                    item['validity'] = validity
                    result_history[tier_list[api_flag]] = (item['execute'] if task_name == "MATH" else item['pred_idx'], validity)

            result_history = build_result_history(result_history, api_trajectory[-1], tier_list)
            metadata = build_metadata(result_history, tier_list)
            
        else:
            min_tier = starter.Starter(question, vectorstore)
            api_flag = tier_list.index(min_tier)

            while validity != 'yes':
                api_trajectory.append(tier_list[api_flag])
                item = base_module.Generator(item, tier_list[api_flag], task_name)
                
                if tier_list[api_flag] != tier_list[-1]: 

                    if tier_list[api_flag] == tier_list[0]:
                        item, validity = base_module.Judge(item, tier_list[api_flag+1], task_name)
                        result_history[tier_list[api_flag]] = (item['execute'] if task_name == "MATH" else item['pred_idx'], validity)
                        api_flag +=1
                    else: 
                        item, validity = base_module.Judge(item, tier_list[api_flag], task_name)
                        result_history[tier_list[api_flag]] = (item['execute'] if task_name == "MATH" else item['pred_idx'], validity)
                        api_flag +=1 
                    
                else:
                    validity = 'yes'
                    item['validity'] = validity
                    result_history[tier_list[api_flag]] = (item['execute'] if task_name == "MATH" else item['pred_idx'], validity)
            
            result_history = build_result_history(result_history, api_trajectory[-1], tier_list)
            metadata = build_metadata(result_history, tier_list)
        
        item['API trajectory'] = api_trajectory
        item['iteration'] = len(api_trajectory)
        item['history'] = result_history
        vectorstore.add_texts([question], metadatas=[metadata])
        final_result.append(item)
        idx +=1
                               
    return final_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-AT: MCQA")

    parser.add_argument("--Tier_list",
                        type=str,
                        default="gpt-4o-mini,gpt-4o,o1-mini,o1",
                        help="Comma-separated list of values")
    
    parser.add_argument("--top_k",
                        type=int,
                        default=5,
                        help="top-k history")
    
    parser.add_argument("--threshold",
                        type=float,
                        default=0.7,
                        help="threshold for the starter")
    
    parser.add_argument("--lambda_",
                        type=float,
                        default=5.0,
                        help="lambda for the accuracy estimator")

    parser.add_argument("--acc_list",
                        type=str,
                        default="0.572,0.679,0.699,0.803",
                        help="Benchmark acc for the accuracy estimator")   

    parser.add_argument("--base_path",
                        type=str,
                        default="/home/injae/LLM-AT/result/MCQA/",
                        help="Base path for save the project")
    
    parser.add_argument("--dataset_name",
                        type=str,
                        default='MCQA',
                        help="Dataset")
    
    parser.add_argument("--save",
                        type=bool,
                        default=True,
                        help="Whether to save the final result")

    args = parser.parse_args()

    tier_list = args.Tier_list.split(",") if args.Tier_list else []
    dataset_name = args.dataset_name
    top_k = args.top_k
    threshold = args.threshold
    lambda_ = args.lambda_
    base_path = args.base_path
    acc_list = [float(x) for x in args.acc_list.split(",")] if args.acc_list else []
    task_name = args.dataset_name
    save = args.save

    embedding_model = HuggingFaceEmbeddings(
        model_name= 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', 
        model_kwargs = {'device':'cuda'},
        encode_kwargs= {'normalize_embeddings': True})
        
    vectorstore = Chroma(
        embedding_function=embedding_model,
        relevance_score_fn= cosine_similarity,
        collection_metadata={"hnsw:space": "cosine"}
        )

    dataset = json.load(open(f'/home/injae/LLM-AT/dataset/{dataset_name}/sample/{dataset_name}.json')) 
    for n, item in enumerate(dataset, start=1):
        item['uid'] = n

    final_result = LLM_AT(dataset, tier_list, top_k, threshold, lambda_, acc_list, vectorstore, task_name)
    text, final_acc, wrong = MCQA_evaluator(final_result)
    total_cost, total_time = cost_evaluator(final_result)

    if save:
        save_path = base_path + f'MCQA_LLMAT_{threshold}_top-{top_k}_{lambda_}.json'    
        with open(save_path, "w") as save:
            json.dump(final_result, save, indent=4)


    print(f'====== LLM-AT: {task_name} ======')
    print(f'Accuracy: {final_acc} % ({text})')
    print(f'Total Cost: $ {total_cost}')
    print(f'Total Execution Time: {total_time} minutes')