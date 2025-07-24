from tqdm import tqdm
import json
import sys
import argparse
sys.path.append('/home/injae/LLM-AT')
from utils.simulation_util import initial_process, build_metadata
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.Module import S_Module
from utils.model_util import cost_evaluator
from utils.math_equivalence import MATH_evaluator
from langchain_chroma import Chroma


def simulation(dataset, tier_list: list, data_pool: list, top_k: int, th: float, lambda_: float, acc_list: list, vectorstore, task_name):

    idx = 1
    acc_T_saver, acc_F_saver = {}, {}
    for t, tier in enumerate(tier_list):
        acc_T_saver[tier] = acc_list[t]
        acc_F_saver[tier] = 1-acc_list[t]

    final_result = []
    starter = S_Module(tier_list, top_k, th, lambda_, acc_list)
    
    for item in tqdm(dataset):
        saver = {}
        api_trajectory = []
        api_model = tier_list[0]
        api_flag = 0

        uid = item['uid']
        question = item['problem']
        saver['uid'] = uid
        saver['problem'] = item['problem']
        saver['level'] = item['level']
        saver['type'] = item['type']
        saver['solution'] = item['solution']

        if idx <= top_k:
            saver, result_history = initial_process(item, saver, api_model, api_flag, tier_list, api_trajectory, data_pool, task_name)

            # metadata = correctness pesudo-label (e.g.,{'gpt-4o-mini': 'finish', 'gpt-4o': 'finish', 'o1-mini': 'finish', 'o1': 'finish'})
            # finish -> correct, continue -> incorrect
            metadata = build_metadata(result_history, tier_list)

        else:
            min_tier = starter.Starter(question, vectorstore)

            if min_tier != tier_list[0]:
                min_index = tier_list.index(min_tier)   
                api_model = min_tier 
                item_= [data for data in data_pool[min_index] if data['uid']==uid][0]

                saver, result_history = initial_process(item_, saver, api_model, min_index, tier_list, api_trajectory, data_pool, task_name)
                metadata = build_metadata(result_history, tier_list) 
            else:
                min_index = 0
                saver, result_history = initial_process(item, saver, api_model, api_flag, tier_list, api_trajectory, data_pool, task_name)
                metadata = build_metadata(result_history, tier_list)
        
        vectorstore.add_texts([question], metadatas=[metadata])  
        final_result.append(saver)
        idx +=1 

    return final_result



if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="LLM-AT Simulation: MATH simulation")

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
                        default="/home/injae/LLM-AT/result/MATH/",
                        help="Base path for save the project")
    
    parser.add_argument("--dataset_name",
                        type=str,
                        default='MATH',
                        help="Dataset")
    
    parser.add_argument("--save",
                        type=bool,
                        default=True,
                        help="Whether to save the final result")

    args = parser.parse_args()

    tier_list = args.Tier_list.split(",") if args.Tier_list else []
    top_k = args.top_k
    threshold = args.threshold
    lambda_ = args.lambda_
    base_path = args.base_path
    acc_list = [float(x) for x in args.acc_list.split(",")] if args.acc_list else []
    task_name = args.dataset_name
    save = args.save

    data_pool = []
    for i, model in enumerate(reversed(tier_list), start=1):
        globals()[f'{model}'] = json.load(open(f'/home/injae/LLM-AT/dataset/MATH/sample/LLMs/{model}_POT.json'))
        data_pool.append(globals()[f'{model}'])
    
    data_pool.reverse()
    init_dataset = data_pool[0]

    embedding_model = HuggingFaceEmbeddings(
        model_name= 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', 
        model_kwargs = {'device':'cuda'},
        encode_kwargs= {'normalize_embeddings': True})
        
    vectorstore = Chroma(
        embedding_function=embedding_model,
        relevance_score_fn= cosine_similarity,
        collection_metadata={"hnsw:space": "cosine"}
        )

    print(f"========= threshold: {threshold}  & lambda: {lambda_}=========")
    simulation_result = simulation(init_dataset, tier_list, data_pool, top_k=top_k, th=threshold, lambda_ = lambda_, acc_list=acc_list, vectorstore=vectorstore, task_name=task_name)
    text, final_acc, wrong = MATH_evaluator(simulation_result)
    total_cost, total_time = cost_evaluator(simulation_result)

    if save:
        save_path = base_path + f'MATH_LLMAT_{threshold}_top-{top_k}_{lambda_}_00.json'    
        with open(save_path, "w") as save:
            json.dump(simulation_result, save, indent=4)

    print(f'====== LLM-AT Simulation: {task_name} ======')
    print(f'Accuracy: {final_acc} % ({text})')
    print(f'Total Cost: $ {total_cost}')
    print(f'Total Execution Time: {total_time} minutes')

    
    



