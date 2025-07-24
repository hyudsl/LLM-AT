import re
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
   
    string = string.replace("\n", "")

    string = string.replace("\\!", "")

    string = string.replace("\\\\", "\\")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")
    
    string = _remove_right_units(string)

    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)

    string = string.replace(" ", "")
    string = string.replace("\\", "")
    string = string.replace("(", "")
    string = string.replace(")", "")
    
    string = _fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2
    
def remove_boxed(s):
    left = "\\boxed{" 
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
    

def last_boxed_only_string(string):   
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def extract_numbers_and_letters(input_string):
    numbers = re.findall(r'\d+', input_string)  
    letters = re.findall(r'[xy]', input_string) 
    return numbers, letters
    

def convert_and_evaluate_fraction(input_string):

    modified_string = re.sub(r'\\(?:dfrac|frac)\{(\d+)\}\{(\d+)\}', r'\1/\2', input_string)
    
    try:
        result = str(round(eval(modified_string),2))
        return result
    except Exception as e:
        numbers = re.findall(r'\d+', input_string)  
        letters = re.findall(r'[xy]', input_string)
        return numbers
    

def convert_to_float(value):
    if value is None:
        value = "-"
    value = value.replace('[', '').replace(']', '').replace('\\!', '').replace(',', '')
    value = value.replace('[','')
    value = value.replace(']', '')
    try:
        value = round(float(value),2)
        return str(value)
    except ValueError:

        value = convert_and_evaluate_fraction(value)
        return value

def MATH_evaluator(data:list):
    wrong = []
    correct = 0
    total = 0

    if len(data) == 0:
        return "", ""
    else:
        for item in data:        
            model_output = item['execute']
            solution = item['solution']
            prob_type = 'a' #item['type']
            level =  'b'#item['level']

            match = re.search(r':\*\*|:', model_output)
            if match:
                index = match.start()
                model_output = model_output[index+1:]
            else:
                model_output = model_output

            model_output = convert_to_float(model_output)
            if type(model_output) == 'str':
                model_output = convert_and_evaluate_fraction(model_output)


            try:
                level = int(level.split("Level ")[1])
            except:
                level = None

            answer = remove_boxed(last_boxed_only_string(solution))
            answer = convert_to_float(answer)

            if type(answer) == 'str':
                answer = convert_and_evaluate_fraction(answer)

            try:
                equiv = is_equiv(model_output, answer)
            except:
                equiv = False

            if equiv:
                correct += 1
            else:
                wrong.append(item)

            total += 1
            texts = f'{correct}/{total}'
            acc =  round(correct/total, 3)*100

        return texts, acc, wrong