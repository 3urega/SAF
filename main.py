from logger import logging_formater
import logging
from models import LinearModel, MLModel
from utils import preprocess
import pandas as pd
import matplotlib.pyplot as plt

MODEL_TO_USE = "Linear" #Linear/ML

def setup_logger():
    handler = logging.StreamHandler()
    handler.setFormatter(logging_formater.ColoredFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


#Usage example. In production you would want to accumulate seen points
#up to the next irrigation and get the current step just by adding 1 each time.

if __name__ == "__main__":
    
    setup_logger()
    logger = logging.getLogger()
    
    model = None
    
    if MODEL_TO_USE == "Linear":
        model = LinearModel()
        logger.info("Selected linear model")
    elif MODEL_TO_USE == "ML":
        model = MLModel()
        logger.info("Selected ML model")
    else:
        logger.ERROR("Incorrect model selected")
        exit()
        
    model.load_plain_model("models/weights/XGBoost_plain_classifier.joblib")
    
    df = pd.read_csv("data/1082-Device-Data-Fix.csv")
    df = preprocess.get_clean_df(df) #Here use whatever function needed to correctly format the data
    df = model.process_dataframe(df)
    
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

    
        
    