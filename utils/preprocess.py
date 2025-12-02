import pandas as pd
import numpy as np
import xgboost as xgb
import re

def get_clean_df(df : pd.DataFrame) -> pd.DataFrame:
    """
    Gets a clean and ordered DataFrame
    """
    
    df.drop(columns=["sensor.deviceSensorid", "position", "sensor.idDecagon", "variable.default_value_name"], inplace=True)
    
    mask = df["variable.name"].str.match("irrigation*")
    df.loc[mask, "depth"] = 0
    
    df["variable_label"] = df["variable.name"] + "_" + df["depth"].astype(int).astype(str)
    df["date"] = pd.to_datetime(df["date"], format="%b %d %Y @ %H:%M:%S.%f")
    pivoted = df.pivot_table(index="date", columns="variable_label", values="variable.normalized_value", aggfunc="first").reset_index()
    pivoted.columns.name = None
    pivoted.sort_values(by="date", inplace=True)
    
    return pivoted

def merge_pred_with_prob(X, y, p=0.2):
    
    swap_mask = np.random.rand(len(X)) < p
    
    y_new = pd.DataFrame(data=y, columns=["soil_moisture_next"])
    y_new["soil_moisture_next"] = y_new["soil_moisture_next"].shift(1)
    
    condition = X["steps_from_peak"] != 0
    
    swap_mask = swap_mask & condition
    
    X_res = X.copy()
    X_res.loc[swap_mask, "soil_moisture_40"] = y[swap_mask]
    
    print(f"Changed rows : {np.sum(swap_mask)}")
    
    return X_res

def merge_diff_with_prob(X, y, p=0.2):
    
    swap_mask = np.random.rand(len(X)) < p
    
    y_new = pd.DataFrame(data=y, columns=["soil_moisture_next"])
    
    y_new["soil_moisture_next"] = y_new["soil_moisture_next"] + X["t2"]
    y_new["soil_moisture_next"] = y_new["soil_moisture_next"].shift(1)
    
    condition = X["next_step"] >= 4
    
    swap_mask = swap_mask & condition
    
    X_res = X.copy()
    X_res.loc[swap_mask, "t2"] = y[swap_mask]
    
    print(f"Changed rows : {np.sum(swap_mask)}")
    
    return X_res

def format_XGBoost(input_vec):
    
    input_names = ["t0", "t1", "t2", "next_step", "hour_s", "hour_c", "season_autumn", "season_spring", "season_summer"]
    input_vec = pd.DataFrame(data=input_vec, columns=input_names)
    input_vec = xgb.DMatrix(input_vec)
    return input_vec

def format_GradientBoost(input_vec):
    input_names = ["soil_moisture_40", "steps_from_peak", "hour"]
    input_vec = pd.DataFrame(data=input_vec, columns=input_names)
    return input_vec
    
def get_capacitancy_points(df : pd.DataFrame):
    
    difference_df = df.copy()

    difference_df['difference'] = np.concat([[np.nan], np.diff(difference_df['soil_moisture_40'])])
    difference_df['difference2'] = np.concat([[np.nan], np.diff(difference_df['difference'])])

    gradient_marker = []
    up_found = False
    down_found = False

    capacitancy_points = []

    found = 0
    beta = 0.3

    for row in difference_df.iloc[1:].itertuples(index=True):
        dx = row.difference
        down = row.irrigation_volume_0 == 0 and dx < 0
        if not up_found and row.irrigation_volume_0 > 0:
            up_found = True
            continue
        if up_found and not down_found and down:
            down_found = True
            value = row.difference2
            found = 0
            continue
        if up_found and down_found:
            prev_value = value
            value = row.difference2
            
            if (prev_value < 0 and value >= 0) and found == 0:
                found += 1
            elif found > 0 and value >= 0:
                found += 1
            else:
                found = 0
                
            if found == 3:
                gradient_marker.append(row.date)
                up_found = False
                down_found = False
                if len(capacitancy_points) > 0:
                    capacitancy_points.append((row.date, beta*capacitancy_points[-1][1] + (1 - beta)*row.soil_moisture_40))
                else:
                    capacitancy_points.append((row.date, row.soil_moisture_40))
                
    capacitancy_frame = pd.DataFrame(capacitancy_points, columns=["date", "capacitancy"])
    
    return capacitancy_frame