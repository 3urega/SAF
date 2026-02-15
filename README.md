# SAF

The report with the results and insights obtained from this project can be found [here](SAF_report.pdf)

## Installation
Just need to do:

```
pip install -r requirements.txt
```


## Structure

The structure of the repository is the next:

```
Saf/
├── logger                          <- Logger formater for better logging
├── models                          <- Classes and weights of the proposed models
│   └── weights                     <- Saved weights (well, complete models I think actually)
├── notebooks                       <- Notebooks used for development. Need to be moved to root folder to work
├── plots                           <- Obtained plots during development.
│   ├── animations                  <- Animations obtained for easy visual understanding of models.
│   ├── errors                      <- Error evolution plots
│   └── preds                       <- Prediction example plots
├── results                         <- Errors extracted from the models
├── trainers                        <- Deprecated. Only needed for some dev. notebook
├── utils                           <- Utility functions shared across modules
├── compute_mean_losses_linear.py   <- Dev. script used to compute losses for the linear model
├── compute_mean_losses_ML.py       <- Dev. script used to compute losses for the ML model
├── preprocess_csv.py               <- Script to clean the SAF csv so they can be read by pandas
└── main.py                         <- Usage example of final refactored classes
```

The current main, without changing anything, expects the data to be in a folder called data.

## Expected input data

The input DataFrames are expected to have at least the following columns:

| Column Name | Data type | Description |
|-------------|-----------|-------------|
| `date` | `pd.Timestamp` | The timestamp at which the moisture was measured |
| `soil_moisture_40` | `float` | The value of soil moisture at depth 40 (cm I think)

Both models and the capacitance detector work with soil moistures at depth 40, since it seem to be the one behaving the best. If other soil moisture depths wanted to be used instead, you would need to change which soil moisture is the model trained on. It shouldn't be too difficult to do within the classes. Otherwise, you can allways change the name of the moisture you want to train to soil_moisture_40 and it will work seamesly. However, this depth is the recomended use case, since it has been developed with this data.