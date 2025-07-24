from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from accelerate import Accelerator, PartialState
from openai import OpenAI
import torch
import time
import os

# os.environ['OPENAI_API_KEY'] = 'your OpenAI API key'
client = OpenAI()

def load_model(huggingface_model, llm_type):
    """
    huggingface_model: bool 
    llm_type: str - llama or qwen
    """
    if not huggingface_model:
        return None, None 

    if "llama" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', 
                                                     torch_dtype=torch.float16, 
                                                     device_map='auto')
    elif "Qwen" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B-Instruct')
        model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-14B-Instruct', 
                                                     torch_dtype=torch.float16, 
                                                     device_map='auto')
    else:
        raise ValueError(f"Unsupported Hugging Face model type: {llm_type}")

    return tokenizer, model


def llama_generate(input_prompt, max_new_tokens, llama_tokenizer, llama):
    input_ids = llama_tokenizer(input_prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to('cuda')
    prompt_len = len(input_ids[0])
    llama_tokenizer.pad_token = "<pad>" 


    terminators = [
    llama_tokenizer.eos_token_id,
    llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    start_time = time.time()
    generation_config = GenerationConfig(
        do_sample = False,
        num_beams = 1,
        temperature = 0.0,
        top_p = 1.0,
        max_new_tokens = max_new_tokens,
        eos_token_id = terminators,
        pad_token_id = llama_tokenizer.eos_token_id
    )

    with torch.no_grad():
        generation_output = llama.generate(
            input_ids = input_ids,
            generation_config = generation_config
        )
        result = generation_output[0][prompt_len:]
        response = llama_tokenizer.decode(result, skip_sepcial_tokens=True)
    
    end_time = time.time()
    
    latency = end_time -start_time
    generated_tokens_count = len(result)

    return response, latency, prompt_len, generated_tokens_count


def Qwen_generate(input_prompt, max_new_tokens, Qwen_tokenizer, Qwen):
    input_prompt = input_prompt + "Please add an end token by appending a \\n to the previously generated text."
    input_ids = Qwen_tokenizer(input_prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to('cuda')
    prompt_len = len(input_ids[0])
   
    messages = [
        {'role': "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant"},
        {'role': "user", "content": input_prompt}
    ]

    text = Qwen_tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt= True
    )
    start_time = time.time()

    model_inputs = Qwen_tokenizer([text], return_tensors='pt').to(Qwen.device)
    generated_ids = Qwen.generate(
            **model_inputs,
            max_new_tokens = max_new_tokens
        )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip (model_inputs.input_ids, generated_ids)]
    response = Qwen_tokenizer.batch_decode(generated_ids, skip_sepcial_tokens=True)[0]
    
    end_time = time.time()
    
    latency = end_time -start_time
    generated_tokens_count = len(generated_ids[0])

    return response, latency, prompt_len, generated_tokens_count


def gpt_generate(llm_api, input_prompt, max_tokens):
    
    content_message= " " 

    start_time = time.time()
    completion = client.chat.completions.create(
    model= llm_api, 
    messages=[
            {"role": "system", "content": content_message},
            {"role": "user", "content": input_prompt}
        ],
    top_p = 1.0,
    frequency_penalty= 0,
    presence_penalty= 0,
    max_tokens=max_tokens,
    )

    response = completion.choices[0].message.content
    end_time = time.time()
    
    latency = end_time -start_time
    input_token = completion.usage.prompt_tokens
    output_token = completion.usage.completion_tokens
    
    return response, latency, input_token, output_token

def o1_generate(llm_api, input_prompt):
    
    start_time = time.time()
    completion = client.chat.completions.create(
    model= llm_api, 
    messages=[
            {"role": "user", 
            "content": input_prompt}
    ]
    )

    response = completion.choices[0].message.content
    end_time = time.time()
    
    latency = end_time -start_time
    input_token = completion.usage.prompt_tokens
    output_token = completion.usage.completion_tokens
    
    return response, latency, input_token, output_token


def LLM(prompt:str, llm_type:str, max_token: int, tokenizer=None, model=None, huggingface_model=False):

    if huggingface_model:
        if model is None or tokenizer is None:
            raise ValueError("Tokenizer and model must be provided when huggingface_model is True.")
        
        if 'llama' in llm_type:
            return llama_generate(prompt, max_token, tokenizer, model)
        
        elif "Qwen" in llm_type:
                return Qwen_generate(prompt, max_token, tokenizer, model)
        
        elif 'gpt' in llm_type:
            return gpt_generate(llm_type, prompt, max_tokens= max_token)
                
        elif 'o1' in llm_type:
            return o1_generate(llm_type, prompt)

    elif 'gpt' in llm_type:
        return gpt_generate(llm_type, prompt, max_tokens= max_token)
    
    elif 'o1' in llm_type:
        return o1_generate(llm_type, prompt)


def cost_calculator(llm_type: str, input_tok: int, output_tok: int):

    # cost is as of 2025-1-17 

    if 'Qwen' in llm_type or 'llama' in llm_type:
        # open-source model
        input_ = ((input_tok/1000000) * 0)
        output_ = ((output_tok/1000000) * 0)
        return input_ + output_

    elif 'gpt-4o-mini' == llm_type:
        input_ = ((input_tok/1000000) * 0.15)
        output_ = ((output_tok/1000000) * 0.6)
        return input_ + output_
    
    elif 'gpt-4o' == llm_type:
        input_ = ((input_tok/1000000) * 2.5)
        output_ = ((output_tok/1000000) * 10)
        return input_ + output_

    elif 'o1-mini' == llm_type:
        input_ = ((input_tok/1000000) * 3)
        output_ = ((output_tok/1000000) * 12)
        return input_ + output_
    
    elif 'o1' == llm_type:
        input_ = ((input_tok/1000000) * 15)
        output_ = ((output_tok/1000000) * 60)
        return input_ + output_
    

def cost_evaluator(data):
    total_cost, total_latency = 0,0
    total_cost = sum(item['g_cost'] + item['j_cost'] for item in data)
    total_latency = sum(item['g_latency'] + item['j_latency'] for item in data)

    return round(total_cost, 3), round((total_latency/60), 3)