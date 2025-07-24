import func_timeout
import re

def safe_execute(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x, locals())
            locals_ = locals()
            if keys is None:
                an = locals_.get('answer', None)
            else:
                an = [locals_.get(k, None) for k in keys]
            return an, "Done"
        except BaseException as e: # jump wrong case
            return None, repr(e)

    try:
        an, report = func_timeout.func_timeout(10, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        an = None
        report = "TimeoutError: execution timeout"

    return str(an), str(report)


def remove_comment(code):  
    code = code.split("\n")
    code = [line for line in code if not line.startswith(("#", '```', 'Code:'))] 
    code = [line for line in code if line.strip() != ""]
    return "\n".join(code)


def cut_comment(code):  
    cut_code = ''
    code = re.split(r'\\n|\n', code)
    for line in code:
        line = line.replace('<|im_end|>', '')
        if 'answer =' in cut_code:
            break
        if line != ' ':
            cut_code += line+'\n'
    return cut_code


def PoT_prompt(prompt: str, example: dict) -> str:
    """Creates a program-of-thought prompt given a single example."""
    prompt_ = prompt
    prompt_ += f"\nQuestion: {example['problem']}\nCode:"
    
    return prompt_


def PoT_EVAL_prompt(prompt: str, example: dict, llm_api: str) -> str:
    """Creates a program-of-thought prompt given a single example."""
    prompt_ = prompt
    prompt_ += f"\nQuestion: {example['problem']}\nCode:\n{example['prediction']}\nexecute: {example['execute']}\nreport: {example['report']}" #\ncurrent LLM api: {llm_api}"
    
    return prompt_

