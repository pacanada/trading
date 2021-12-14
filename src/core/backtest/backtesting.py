import numba
from typing import List, Tuple
import numpy as np
@numba.jit(nopython=True, parallel=True)
def backtest_numba(
    indexes: np.array,
    open_values: np.array,
    preds_values: np.array, 
    take_profit_pct: float = 0.04,
    stop_loss_pct:float=0.04,
    lim_pred_sell: float=3,
    lim_pred_buy:float=-3,
    security_factor:float=0.0024,
    fee: float = 0.0016,
    shift: int = 1
)->List[Tuple[int, int, float, int]]:
    list_action = []
    # buy 0 sell 1
    next_action: int = 0
    previous_price: float = 0
    # comments 0= predicted buy, 1 take_profit, 2 stop_loss
    
    for index, (i, pred_value, open_value) in enumerate(zip(indexes, preds_values, open_values)):
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
    if len(list_action)>0:
        if (indexes[-1] == list_action[-1][0]) and (list_action[-1][1]==1):
            # remove if there is a last sell in the last interval because we do not now the next open price
            # weird error. Example with bateur where the cost using the open price is set to 500 for no reason
            list_action.remove(list_action[-1])
                
        if list_action[-1][1]==0:
            # remove if last is also a buy
            list_action.remove(list_action[-1])

    return list_action
@numba.jit(nopython=True, parallel=True)
def backtest_numba_buy_and_sell(
    indexes: np.array,
    open_values: np.array,
    preds_values: np.array, 
    take_profit_pct: float = 0.04,
    stop_loss_pct:float=0.04,
    security_factor:float=0.0024,
    sell_lim: float = 3,
    buy_lim:float=-3,
    fee: float = 0.0016,
    shift: int = 1
)->List[Tuple[int, int, float, int]]:
    list_action = []
    # buy 0 sell 1
    next_action: int = 0
    previous_price: float = 0
    # comments -1 predicted sell,0= predicted buy, 1 take_profit, 2 stop_loss
    
    for index, (i, pred_value, open_value) in enumerate(zip(indexes, preds_values, open_values)):
        # sell 1, buy 0
        if index == len(preds_values)-1:
            continue
        if pred_value<buy_lim and not next_action:
            cost = -open_values[index+shift]*(1+fee+security_factor)
            list_action.append((i, next_action, cost, 0))
            previous_price = open_value
            next_action = 1
        elif pred_value>sell_lim and next_action:
            cost = open_values[index+shift]*(1-fee-security_factor)
            list_action.append((i, next_action, cost, -1))
            previous_price = open_value
            next_action = 0
            
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
    if len(list_action)>0:
        if (indexes[-1] == list_action[-1][0]) and (list_action[-1][1]==1):
            # remove if there is a last sell in the last interval because we do not now the next open price
            # weird error. Example with bateur where the cost using the open price is set to 500 for no reason
            list_action.remove(list_action[-1])
                
        if list_action[-1][1]==0:
            # remove if last is also a buy
            list_action.remove(list_action[-1])

    return list_action

def backtest_pure_python(
    indexes: np.array,
    open_values: np.array,
    preds_values: np.array, 
    take_profit_pct: float = 0.04,
    stop_loss_pct:float=0.04,
    lim_pred_sell: float=3,
    lim_pred_buy:float=-3,
    security_factor:float=0.0024,
    fee: float = 0.0016,
    shift: int = 1
)->List[Tuple[int, int, float, int]]:
    list_action = []
    # buy 0 sell 1
    next_action: int = 0
    previous_price: float = 0
    # comments 0= predicted buy, 1 take_profit, 2 stop_loss
    
    for index, (i, pred_value, open_value) in enumerate(zip(indexes, preds_values, open_values)):
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
    if (indexes[-1] == list_action[-1][0]) and (list_action[-1][1]==1):
        # remove if there is a last sell in the last interval because we do not now the next open price
        # weird error. Example with bateur where the cost using the open price is set to 500 for no reason
        list_action.remove(list_action[-1])
            
    if list_action[-1][1]==0:
        # remove if last is also a buy
        list_action.remove(list_action[-1])

    return list_action