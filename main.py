from logger import logging_formater
import logging
from models import LinearModel, MLModel, CapacitanceDetector
from utils import preprocess
import pandas as pd
import matplotlib.pyplot as plt

"""
Usage example. In production you would want to accumulate seen points
and get the current step and date just by counting number of points 
and knowing the first date. With that you avoid redundant communications 
and save time and resources. When a new irrigation is done, the points are
reseted and you get again a new initial date to account for the irrigation
time. Could just stop communicating new points when irrigating and make
model reset after some time to avoid overheads, but that would make the
system non-reliable against possible communication delays, so I would
not recommend that.
"""

MODEL_TO_USE = "ML" #Linear/ML

def setup_logger():
    handler = logging.StreamHandler()
    handler.setFormatter(logging_formater.ColoredFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


if __name__ == "__main__":
    
    setup_logger()
    logger = logging.getLogger()
    
    df = pd.read_csv("data/1082-Device-Data-Fix.csv")
    df = preprocess.get_clean_df(df) #Here use whatever function needed to correctly format the data
    
    model = None
    
    if MODEL_TO_USE == "Linear":
        #This training takes quite a lot of time since it searches for the best possible
        #parameters, so I'd recommend to train once and load everytime after that.
        model = LinearModel()
        model.load_plain_model("models/weights/XGBoost_plain_classifier.joblib")
        logger.info("Selected linear model")
        
    elif MODEL_TO_USE == "ML":
        #This gets trained really fast, so it is not really needed to store a model
        #however, it can be saved and loaded if there is the need.
        model = MLModel()
        model.train(df, save_model=False)
        logger.info("Selected ML model")
        
    else:
        logger.ERROR("Non-existing model selected")
        exit()
        
    
    #Example date choosen for great visualization
    start_date = pd.to_datetime("2024-06-06 12:00:00")
    end_date   = pd.to_datetime("2024-06-11 05:00:00")
    
    #Get points after irrigation and last date before prediction
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    mask = df["irrigation_volume_0"] == 0
    first_point = mask[mask].index[0]
    df = df.loc[first_point:].copy()
    
    previous_values = df["soil_moisture_40"].iloc[:10].values
    current_date = df["date"].iloc[9]
    
    predictions = model.predict_steps(previous_values, current_date, 10, 100)
    
    correct_values = df["soil_moisture_40"].iloc[10:110].values
    
    steps = [i for i in range(110)]
    
    plt.scatter(steps[:10], previous_values, color="blue", label="Seen points")
    plt.plot(steps[10:], predictions, color="red", label="Predicted")
    plt.plot(steps[10:], correct_values, color="green", label="Real future values")
    plt.legend(loc="upper right")
    plt.ylabel("Soil moisture")
    plt.xlabel("Steps")
    plt.ylim(0.20, 0.36)
    plt.show()
    
    #Here we find capacitances. An anomaly detector or something like
    #that should be added into this in production, so big outliers do 
    #not hurt the user visualizaton when using it for normalization.
    capacitance_detector = CapacitanceDetector()
    capacitances = capacitance_detector.detect_capacitances(df)
    plt.plot(df["date"], df["soil_moisture_40"], label="Soil moisture")
    
    plt.axvline(capacitances["date"].iloc[0], color="red", label="Capacitance")
    for row in capacitances.iloc[1:].itertuples():
        plt.axvline(row.date, color="red")

    plt.legend(loc="upper right")
    plt.ylabel("Soil moisture")
    plt.xlabel("Date")
    plt.ylim(0.20, 0.36)
    plt.show()

    
        
    