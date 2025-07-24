import random
from collections import namedtuple
import tqdm as tqdm
import re
import string


def chain_of_thought_prompt(question:str,example: dict) -> str:

    alphabet = string.ascii_uppercase

    """Creates a chain-of-thought prompt given a single example."""
    prompt = f"Here is a question. An explanation is given before the final answer. Answer the question yourself, giving your reasoning beforehand.\n"
    prompt += f"Be sure to generate the final answer in 'The correct answer is' format.\n"
    prompt += f"Question: {question}"
    prompt += f"\nChoices:"
    
    for i, key in enumerate(example['Choices']):
    
        prompt += f"\n({alphabet[i]}) {example['Choices'][key]}"

    prompt += "\nThe output format must strictly follow three inference statements and The correct answer is: (answer index (e.g., A) here). \nGive step by step reasoning in a maximum of three sentences before your answer. Do not exceed three sentences."
    return prompt


def chain_of_thought_Abstain_prompt(question:str,example: dict) -> str:

    alphabet = string.ascii_uppercase

    """Creates a chain-of-thought prompt given a single example."""
    prompt = f"Review the following math problem and first determine whether you can solve it accurately.\n"
    prompt += f"If you are confident that you can solve the problem, answer the question yourself, providing logical reasoning beforehand.\n"
    prompt += f"Be sure to generate the final answer in 'The correct answer is' format.\n"
    prompt += f"Question: {question}\nChoices:"
    
    for i, key in enumerate(example['Choices']):
    
        prompt += f"\n({alphabet[i]}) {example['Choices'][key]}"

    prompt += f"\nThe output format must strictly follow three inference statements and The correct answer is: (answer index (e.g., A) here). \nGive step by step reasoning in a maximum of three sentences before your answer. Do not exceed three sentences.\n"
    prompt += f"If there is even the slightest doubt about solving the problem correctly, output only “Abstain” and do not generate any additional content.\n\n"
    prompt += f"Follow the instructions and handle the problem appropriately without overestimating your abilities."

    return prompt


def MCQA_EVAL_prompt(prompt: str, question, prediction, example:dict) -> str:
    alphabet = string.ascii_uppercase

    """Creates a program-of-thought prompt given a single example."""
    prompt_ = prompt
    prompt_ += f"\nquestion: {question}\n\nChoices:"
    
    for i, key in enumerate(example['Choices']):
        prompt_ += f"\n({alphabet[i]}) {example['Choices'][key]}"

    prompt_ += f'\n\ngenerated output: {prediction}'    
    
    return prompt_


def extract_correct_answer(text: str) -> str:
    texts = text.split('\n')
    for item in reversed(texts):
        if item != '':
            line = item
            line = line.replace("\\n", "")
            break
        
    match = re.search(r'the correct answer is \(([A-Za-z])\)', line)

    key_flag = False
    if match:
        key_flag = True
        key= match.group(1).replace('(', '').replace(')', '').upper()
        if key not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
            return 'wrong_key'
        else: 
            return key

    if not key_flag:
        line = line.replace(":", "").replace(")", '').replace('(','')
        alternative_match = re.search(r'the correct answer is ([A-Za-z])', line)
        alternative_match2 = re.search(r'the correct answer is \*\*([A-Za-z])\*\*', line)
        if alternative_match:
            key = alternative_match.group(1).replace('(', '').replace(')', '').upper()
            
            if key not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                return 'wrong_key'
            else: 
                return key
        
        elif alternative_match2:
            key = alternative_match2.group(1).replace('(', '').replace(')', '').upper()

            if key not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                return 'wrong_key'
            else: 
                return key
    
    return 'wrong_key'

    
def trim_after_correct_answer(text):
    match = re.search(r'The correct answer is \([A-Z]\)\.', text)
    if match:
        return text[:match.end()] 
    return text 


def cut_response(response: str):
    texts = response.split('\n')
    text = ''
    for line in texts:
        text += trim_after_correct_answer(line) + '\n'
        
        if line.startswith('The correct'):
            break
    return text


def option_change(item):
    options = [item['Choice 1'], item['Choice 2'], item['Choice 3'], item['Choice 4']]
    options_c = {'A':item['Choice 1'], 'B':item['Choice 2'], 'C':item['Choice 3'], 'D':item['Choice 4'], 'wrong_key':'wrong_answer_format'}

    return options, options_c


def index_extractor(item:dict, response:str):
    alphabet = string.ascii_uppercase
    choices_pool = {}
    for i, key in enumerate(item['Choices']):
    
        choices_pool[alphabet[i]] = item['Choices'][key]

    response = response.lower()
    llm_reponse_idx = extract_correct_answer(response)

    if llm_reponse_idx != 'no_match_found' and llm_reponse_idx != 'wrong_key':
        llm_reponse = choices_pool[llm_reponse_idx]
    else:
        gold = item['correct index']
        remaining_keys = [key for key in choices_pool if key != gold]
        random_choice = random.choice(remaining_keys)
        llm_reponse = choices_pool[random_choice]

    return llm_reponse_idx, llm_reponse


def MCQA_evaluator(data:list):
    total = 0
    correct = 0
    wrong = []

    for item in data:
        answer = item['correct index']
        model_output = item['pred_idx']

        if answer.lower() == model_output.lower():
            correct +=1
            total +=1
        else:
            total +=1
            wrong.append(item)
    
    texts = f'{correct}/{total}'
    acc = round(correct/total, 5)*100

    return texts, acc, wrong

