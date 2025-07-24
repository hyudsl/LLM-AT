from utils.model_util import cost_calculator

def upgrade_process(item, api_model, api_state, api_flag, result_history: list, api_trajectory:list, tier_list: list, data_pool: list, task_type:str):
    """
    The inference process that upgrades tiers until a valid answer is generated
    """
    uid = item['uid']
    g_cost = cost_calculator(item['API'], item['g_input_tok'], item['g_output_tok'])
    j_cost = cost_calculator(item['J_API'], item['j_input_tok'], item['j_output_tok'])
    g_latency = item['g_latency'] 
    j_latency = item['j_latency']

    while api_state != 'finish':
        api_flag +=1
        api_model = tier_list[api_flag]
        data = data_pool[api_flag]
        api_trajectory.append(api_model)

        upgrade_item = [up_item for up_item in data if up_item['uid'] == uid][0]
        g_cost += cost_calculator(upgrade_item['API'], upgrade_item['g_input_tok'], upgrade_item['g_output_tok'])
        g_latency += upgrade_item['g_latency']
        
        if api_model != tier_list[-1]:
            j_cost += cost_calculator(upgrade_item['J_API'], upgrade_item['j_input_tok'], upgrade_item['j_output_tok'])
            j_latency += upgrade_item['j_latency']

        # Abstain or not
        Abstain = 'abstain' in upgrade_item['prediction'].lower().replace('\\n', '')
        if Abstain:
            api_state = 'upgrade'
            result_history.append(('abstain', )) 
        else:
            result_history.append((upgrade_item['execute'] if task_type == "MATH" else upgrade_item['pred_idx'], ))
            if upgrade_item['validity'] == 'yes':
                api_state = 'finish'
                result_history[-1] = (upgrade_item['execute'] if task_type == "MATH" else upgrade_item['pred_idx'], 'finish')

        # the last layer or not
        if api_model == tier_list[-1] and api_state != 'finish':
            api_state = 'finish'
            result_history[-1] = (upgrade_item['execute'] if task_type == "MATH" else upgrade_item['pred_idx'], 'finish')

        final_result = {'item':upgrade_item, 'g_cost':g_cost, 'j_cost':j_cost, 
                        'g_latency':g_latency, 'j_latency':j_latency, 'trajectory':api_trajectory}
        
    return final_result, result_history


def initial_process(item:dict, saver:dict, api_model, api_flag, tier_list, api_trajectory, data_pool, task_type):
    Abstain = 'abstain' in item['prediction'].lower().replace('\\n', '')

    if Abstain:
        api_trajectory.append(tier_list[api_flag])
        init_result_history = [('abstain',)] 
        api_state = 'finish' if api_model == tier_list[-1] else 'upgrade'

        final_result, result_history = upgrade_process(item, api_model, api_state, api_flag, init_result_history, api_trajectory, tier_list, data_pool, task_type)

        result_history_dic = dict(zip(api_trajectory, result_history))
        result_history = build_result_history(result_history_dic, api_trajectory[-1], tier_list)

    else: 
        api_state = 'finish' if item['validity'] == 'yes' else 'upgrade' 
        if api_model == tier_list[-1] and api_state != 'finish':
            api_state = 'finish'
        
        api_trajectory.append(tier_list[api_flag])

        if api_state == 'finish':   # The starter’s chosen initial tier immediately generates a valid response
            g_cost = cost_calculator(item['API'], item['g_input_tok'], item['g_output_tok'])
            j_cost = cost_calculator(item['J_API'], item['j_input_tok'], item['j_output_tok']) if api_model != tier_list[-1] else 0
            g_latency = item['g_latency']
            j_latency = item['j_latency'] if api_model != tier_list[-1] else 0
           
            result_history = [('', '')]*len(tier_list)
            for i in range(0, len(tier_list)):                       
                if i == api_flag:
                    result_history[i] = (item['execute'] if task_type == "MATH" else item['pred_idx'], 'finish')
                elif i > api_flag:
                   result_history[i] = (("-", 'finish')) 

            final_result = {'item':item, 'g_cost':g_cost, 'j_cost':j_cost, 'g_latency':g_latency, 
                            'j_latency':j_latency, 'trajectory':api_trajectory}  
        else:
            init_result_history = [(item['execute'] if task_type == "MATH" else item['pred_idx'],)]

            final_result, result_history = upgrade_process(item, api_model, api_state, api_flag, init_result_history, api_trajectory, tier_list, data_pool, task_type)

            result_history_dic = dict(zip(api_trajectory, result_history))
            result_history = build_result_history(result_history_dic, api_trajectory[-1], tier_list)     

    final_saver = build_saver(saver, final_result, task_type) 
    for i, history_item in enumerate(result_history):
        final_saver[tier_list[i]] = history_item

    return final_saver, result_history


def build_result_history(result_dict, finish_api, tier_list): 
    """
    Updates correctness labels based on the result when a starter is used
    > If any lower tier produces the same answer as the final result, mark it as ‘finish’; all higher tiers are marked ‘finish’ by default
    > Leave lower tiers empty if no result is available
    """
    finish_num = tier_list.index(finish_api)                  
    finish_line = result_dict[finish_api][0]                     
    result = [('', '')] * len(tier_list)

    for i, api in enumerate(tier_list):
        if api in result_dict:
            if i < finish_num:
                status = 'finish' if result_dict[api][0] == finish_line else 'continue'
            elif i == finish_num:
                status = 'finish'
                
            result[i] = (result_dict[api][0], status)
        elif i > finish_num:
            result[i] = ('-', 'finish')
    return result


def build_saver(saver, result, task_type):

    if task_type == 'MATH':
        saver['prediction'] = result['item']['prediction']
        saver['execute'] = result['item']['execute']

    elif task_type == 'MCQA':
        saver['prediction'] = result['item']['prediction']
        saver['pred_idx'] = result['item']['pred_idx']
    else:
        raise ValueError(f"Unsupported Task Type: {task_type}")

    saver['g_cost'] = result['g_cost']
    saver['j_cost'] = result['j_cost']
    saver['g_latency'] = result['g_latency']
    saver['j_latency'] = result['j_latency']
    saver['API trajectory'] = result['trajectory']
    saver['iteration'] = len(result['trajectory'])
    return saver


def build_metadata(result_history, tier_list):
    return {tier: result_history[i][1] for i, tier in enumerate(tier_list)}