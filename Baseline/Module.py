import sys
sys.path.append('/home/injae/LLM-AT')
from tqdm import tqdm
from utils.MATH_util import safe_execute, remove_comment, cut_comment, PoT_prompt, PoT_EVAL_prompt
from utils.model_util import LLM, cost_calculator
from utils.Prompt import MATH_POT_INIT, MATH_POT_ABS, MATH_EVAL, MCQA_EVAL
from utils.MCQA_util import  chain_of_thought_prompt, chain_of_thought_Abstain_prompt, cut_response, MCQA_EVAL_prompt, index_extractor


def get_prompt(task_type, llm_type, item):
    if task_type == "MATH":
        prompt_type = MATH_POT_ABS if llm_type == 'gpt-4o-mini' else MATH_POT_INIT
        return PoT_prompt(prompt_type, item)
    else:
        return chain_of_thought_Abstain_prompt(item['question'], item) if llm_type == 'gpt-4o-mini' \
               else chain_of_thought_prompt(item['question'], item)
    
def get_eval_prompt(task_type, item, llm_type):
    if task_type == 'MATH':
        return PoT_EVAL_prompt(MATH_EVAL, item, llm_type)
    else:
        question = item['question']
        prediction = item['prediction'].replace("<|im_end|>", "")
        return MCQA_EVAL_prompt(MCQA_EVAL, question, prediction, item)
    
def process_math_response(item, response):
    response = cut_comment(response)
    clean = remove_comment(response)
    answer, report = safe_execute(clean)
    item['prediction'] = response
    item['execute'] = answer
    item['report'] = report
    return item

def process_mcqa_response(item, response):
    response = response.replace("<|im_end|>", "")
    answer = cut_response(response)
    pred_idx, _ = index_extractor(item, answer)
    item['prediction'] = response
    item['pred_idx'] = pred_idx
    return item

def parse_validity(response):
    response = response.lower()
    if 'yes' in response:
        return 'yes'
    else:
        return 'no'
    

class GJ_Module:
    def __init__(self, model, tokenizer, huggingface_model):
        self.model = model
        self.tokenizer = tokenizer
        self.huggingface_model = huggingface_model

    def Generator(self, dataset, iter_flag, llm_type, task_type):

        for item in tqdm(dataset):

            prompt_ = get_prompt(task_type, llm_type, item)

            g_response, g_latency, g_input_tok, g_output_tok = LLM(
                prompt_,
                llm_type,
                max_token=300,
                tokenizer=self.tokenizer,
                model=self.model,
                huggingface_model=self.huggingface_model 
                )
            g_cost = cost_calculator(llm_type, g_input_tok, g_output_tok)
            
            if task_type == "MATH":
                item = process_math_response(item, g_response)
            else:
                item = process_mcqa_response(item, g_response)

            item['iteration'] = iter_flag
            item['g_latency'] = item.get('g_latency', 0) + g_latency
            item['g_cost'] = item.get('g_cost', 0)+g_cost
    
        return dataset, iter_flag

    def Judge(self, dataset, iter_flag, llm_type, task_type):

        for item in tqdm(dataset):

            Abstain = 'abstain' in item['prediction'].lower().replace('\\n', '')

            if Abstain: 
                validity = 'no'
                j_cost = 0
                j_latency = 0

            else:
                eval_prompt = get_eval_prompt(task_type, item, llm_type)

                response, j_latency, j_input_tok, j_output_tok = LLM(
                    eval_prompt,
                    llm_type,
                    max_token=50,
                    tokenizer = self.tokenizer,
                    model = self.model,
                    huggingface_model = self.huggingface_model  
                    )

                j_cost = cost_calculator(llm_type, j_input_tok, j_output_tok)
                judgement_history = item.get('judgement result', [])
                validity = parse_validity(response)

            item['validity'] = validity
            judgement_history.append(validity)
            item['judgement result'] = judgement_history

            item['j_latency'] = item.get('j_latency', 0) + j_latency
            item['j_cost'] = item.get('g_cost',0)+j_cost

        finish_dataset = [item for item in dataset if item['validity'] == 'yes']
        remain_dataset = [item for item in dataset if item['validity'] != 'yes']

        return remain_dataset, finish_dataset, iter_flag
