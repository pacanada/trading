from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
cpdef backtest_cpp(
    vector[int] indexes,
    vector[float] open_values,
    vector[float] preds_values,
    double take_profit_pct,
    double stop_loss_pct,
    double lim_pred_buy,
    double security_factor,
    double fee,
    int shift):

    cdef list list_action = []
    # buy 0 sell 1
    cdef int next_action = 0
    cdef double previous_price = 0
    # comments 0= predicted buy, 1 take_profit, 2 stop_loss

    cdef int index = 0
    cdef int i
    cdef double pred_value
    cdef double open_value
    cdef double cost 
    
    for i, pred_value, open_value in zip(indexes, preds_values, open_values):
        # sell 1, buy 0
        if pred_value < lim_pred_buy and not next_action:
            cost = -open_values[index+shift]*(1+fee+security_factor)
            list_action.append((i, next_action, cost, 0))
            previous_price = open_value
            next_action = 1
            
        elif (open_value>(1+take_profit_pct)*previous_price) and next_action:
            cost = open_values[index+shift]*(1-fee-security_factor)
            list_action.append((i, next_action, cost, 1))
            previous_price = open_value
            next_action = 0
        elif (open_value<(1-stop_loss_pct)*previous_price) and next_action:
            cost = open_values[index+shift]*(1-fee-security_factor)
            list_action.append((i, next_action, cost, 2))
            previous_price = open_value
            next_action = 0 
        index += 1
    if (indexes[-1] == list_action[-1][0]) and (list_action[-1][1]==1):
        # remove if there is a last sell in the last interval because we do not now the next open price
        # weird error. Example with bateur where the cost using the open price is set to 500 for no reason
        list_action.remove(list_action[-1])
            
    if list_action[-1][1]==0:
        # remove if last is also a buy
        list_action.remove(list_action[-1])

    return list_action


cpdef backtest_cpp_no_list(
    vector[int] indexes,
    vector[float] open_values,
    vector[float] preds_values,
    double take_profit_pct,
    double stop_loss_pct,
    double lim_pred_buy,
    double security_factor,
    double fee,
    int shift):
    """It seems that using list and tuples complicate the process and hinders speed-up"""

    # buy 0 sell 1
    cdef int next_action = 0
    cdef double previous_price = 0
    # comments 0= predicted buy, 1 take_profit, 2 stop_loss

    cdef int index = 0
    cdef int index_action = 0
    cdef int i
    cdef double pred_value
    cdef double open_value
    cdef double cost
    cdef int size = len(indexes) 
    cdef list_action = np.empty([size, 4])  
    
    for i, pred_value, open_value in zip(indexes, preds_values, open_values):
        # sell 1, buy 0
        if pred_value < lim_pred_buy and not next_action:
            cost = -open_values[index+shift]*(1+fee+security_factor)
            list_action[index_action, :] = [i, next_action, cost, 0]
            previous_price = open_value
            next_action = 1
            index_action+=1
            
        elif (open_value>(1+take_profit_pct)*previous_price) and next_action:
            cost = open_values[index+shift]*(1-fee-security_factor)
            list_action[index_action, :] = [i, next_action, cost, 1]
            previous_price = open_value
            next_action = 0
            index_action+=1
        elif (open_value<(1-stop_loss_pct)*previous_price) and next_action:
            cost = open_values[index+shift]*(1-fee-security_factor)
            list_action[index_action, :] = [i, next_action, cost, 2]
            previous_price = open_value
            next_action = 0 
            index_action+=1
        index += 1
    if (indexes[-1] == list_action[-1][0]) and (list_action[-1][1]==1):
        # remove if there is a last sell in the last interval because we do not now the next open price
        # weird error. Example with bateur where the cost using the open price is set to 500 for no reason
        list_action[-1, :] = [0,0,0,0]
            
    if list_action[-1][1]==0:
        # remove if last is also a buy
        list_action[-1, :] = [i, next_action, cost, 2]

    return list_action