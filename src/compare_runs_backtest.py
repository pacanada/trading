#Deserialize
import pickle
import pandas as pd
from src.modules.paths import get_project_root
from src.core.features.utils import feature_pipeline, add_domain_features
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
with open("notebooks/wow_20_domain.pickle" ,'rb') as f:
    model = pickle.load(f)


def backtest(df: pd.DataFrame, model, threshold, target, fee, columns_features, shift_in_preds):
    """"""
    #if target not in df.columns:
    #    raise KeyError(f"{target} not included in columns")
        
    #df_backtesting = df[[target, "open"]].copy()
    df_backtesting = df[["open"]].copy()
    df_backtesting["preds"] = model.predict(df[columns_features])
    
    df_backtesting = add_actions(df_backtesting, threshold, shift_in_preds)
    return evaluate(df_backtesting, fee)
    
    
    
def add_actions(df_backtesting, threshold, shift_in_preds):
    next_action = "buy"
    df_backtesting["action"] = None
    df_backtesting["sum"] = None
    cont = 0
    wait_interval = 0
    print(wait_interval, shift_in_preds)
    for index, row in df_backtesting.iterrows():

        if index == 0:
            continue
        if (-df_backtesting.iloc[index-shift_in_preds]["preds"] > threshold) and (next_action=="buy") and cont>=wait_interval:
            df_backtesting.loc[df_backtesting.index==index, "action"] = "buy"
            next_action = "sell"
            cont = 0
        elif (-df_backtesting.iloc[index-shift_in_preds]["preds"] < -threshold) and (next_action=="sell") and cont>=wait_interval:
            df_backtesting.loc[df_backtesting.index==index, "action"] = "sell"
            next_action = "buy"
            cont=0
        cont+=1
    return df_backtesting

def evaluate(df_backtesting, fee, market_extra=0.0025):
    df_backtesting["sum"] = None
    df_backtesting.loc[(df_backtesting.action=="buy"), "sum"] = -df_backtesting[df_backtesting.action=="buy"].open*(1+fee+market_extra)
    df_backtesting.loc[(df_backtesting.action=="sell"), "sum"] = df_backtesting[df_backtesting.action=="sell"].open*(1-(fee+market_extra))
    if df_backtesting[df_backtesting.action.notnull()].tail(1).action.values[0]=="buy":
        df_backtesting.loc[df_backtesting[df_backtesting.action.notnull()].tail(1).index, ["sum", "action"]] = None
    if df_backtesting[df_backtesting.action.notnull()].tail(1).action.values[0]=="buy":
        raise ValueError("last action cannot be buy")
    n_trades = df_backtesting["sum"].notnull().sum()
    profit = df_backtesting["sum"].sum()/df_backtesting["open"].mean()
    hist_ratio = df_backtesting.tail(1).open.values[0]/df_backtesting.head(1).open.values[0]-1
    return n_trades, profit, hist_ratio
    
cut_off_date ="2021-11-07 21:28:00"
columns_features = [col for col in feature_pipeline(pd.read_csv(get_project_root() / "data" / "historical" / "etheur.csv"), include_target=True, target_list=[10])[0].columns if col.startswith("feature_domain")]

# test real data
#eth
list_results=[]
for pair_name in ["xlmeur", "bcheur","compeur","xdgeur", "etheur", "algoeur", "bateur", "adaeur","xrpeur"]:
    df = pd.read_csv(get_project_root() / "data" / "historical" / f"{pair_name}.csv")
    df = df[df.date>cut_off_date]
    df, _=feature_pipeline(df, include_target=True, target_list=[10, 20])
    for threshold in [.3, .5, .7, .9]:
        n_trades, profit, hist_ratio = backtest(
            df=df,
            model=model,
            threshold=threshold,
            target="target_20",
            fee=0.0016,
            columns_features=columns_features,
            shift_in_preds=1
        )
        print(threshold, pair_name, n_trades, profit, hist_ratio)
        list_results.append((threshold, pair_name, n_trades, profit, hist_ratio))