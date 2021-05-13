# Disaster Response Pipeline Webapp

## Overview

This repo is an extension of [this one](https://github.com/russom/DSND-P3-Disaster-Response-Pipeline), containin code for the creation of a Disaster Response webapp capable of classifying messages from user based on a ML model trained for that purpose.

The code in the original repo contains all the scripts needed to handle the data and train the classifier: it also includes the cose for a webapp capable of running locally on your machine.
The code in this repo focuses only on the webapp part, with the scope of packaging it for a cloaud deployment. The target platform chosen is [Heroku](https://www.heroku.com/).


## Requirements
In order to run the app locally I have prepared an [`environment.yml`](./environment.yml) file to be used to install an environment with [Anaconda](https://www.continuum.io/downloads):

```sh
conda env create -f environment.yml
```

After the installation the environment should be visible via `conda info --envs`:

```sh
# conda environments:
#
dsnd-proj3        /usr/local/anaconda3/envs/dsnd-proj3
...

```

the full list of requirements that get loaded is also visible in [`requirements.txt`](./requirements.txt), that will be used by Heroku.

## Instructions
The code in this repo includes 2 jupyter notebooks (in the [`notebooks`](./notebooks) folder) and a three scripts. 

* The notebooks are provided as guidelines/references for the scripts. They do not need to be executed to run the webapp:
    - [`ETL Pipeline Preparation`](./notebooks/ETL_Pipeline_Preparation.ipynb) documents a step-by-step process to load data from the `.csv` files and save them in an SQL-lite DB;
    - [`ML_Pipeline_Preparation`](./notebooks/ML_Pipeline_Preparation.ipynb) documents a step-by-step process to load data from the DB generated previously and train a classifier on them.

* In order to use the scripts to set up the database and the model from the project's root directory, you'll need to use the following commands and arguments:
    - To run the ETL pipeline that cleans data and stores in database you'll need to run [`process_data.py`](./data/data_scripts/process_data.py):  
        `python data/data_scripts/process_data.py data/data_files/disaster_messages.csv data/data_files/disaster_categories.csv` _`{path to database file}`_;
    - To run the ML pipeline that trains a classifier, saves it in a pickle file and also saves a `.txt` file containing an evaluation report based on [`sklearn classification_report()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) you'll need to run [`train_classifier.py`](./models/models_scripts/train_classifier.py):  
        `python models/models_scripts/train_classifier.py` _`{path to database file}`_ _`{path to model file}`_ _`{path to report file}`_;
    - **Note** that the last argument for the previous script is optional: if you don't define a report file the outcome of the `classification_report()` will be displayed on screen.
        
 
* Finally, to run the webapp executing [`app.py`](./app/app.py) from the root directory you can:
    -  Run the following command, including arguments to point to the files of your choice:  
        `python app/app.py` _`{path to database file}`_ _`{path to model file}`_
    - Simply run:  
        `python app/app.py`   
        In this case the app will use two default file paths: `data/data_db/disaster_responses.db` for the DB, `models/models_files/cv_trained_model.pkl` for the pickle file with the classifier.

* To see the webapp in your browser go to http://0.0.0.0:3001/ . From the page you'll be able to:
    - See some stastics regarding the dataset;
    - Type a new message and run the classifier against it.  
    
## Results
* A database containing the processed values is available in [`data/data_db`](./data/data_db) as `disaster_responses.db`. 
    - The DB includes a single table called `DisasterResponses`. This is the name of the table expected by the code at the moment: to see where the DB is generated check the `save_data` function in [`process_data.py`](./data/data_scripts/process_data.py) (lines 74-94). To see where the data are read in check the `load_data` function in [`train_classifier.py`](./models/models_scripts/train_classifier.py) (lines 25-56).  

* A pickle file containing a dictionary that includes a model and the datasets used to train/test is available separatedly [here](https://drive.google.com/file/d/1laeKEC0yin0gqBFHb2-mLAgR308N5dsw/view?usp=sharing) as `cv_trained_model.pkl`, given its size (~1GB).
    - The code is expecting to load from the pickle file a dictionary with specific fields:  
        `X_train`: The dataset of features used to train the model.   
        `X_test`: The dataset of features used to train the model.   
        `y_train`: The dataset of labels used to train the model.   
        `y_test`: The dataset of labels used to train the model.   
        `model`: The actual model.   
    - To see how the model is saved check the `save_model` function in [`train_classifier.py`](./models/models_scripts/train_classifier.py) (lines 176-199).  

* An evaluation report for the model is available in [`models/models_files`](./models/models_files) as `cv_training_report.txt`.  

All files where generated with the scripts above, and can be used to run the webapp.  

A **note** on performances: as it can be seen in the `build_model()` function that is part of the [`train_classifier.py`](./models/models_scripts/train_classifier.py) script (lines 87-123), the model to be trained is actually a [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) object, that iterates over a grid of parameters - see lines 111-119:

```
# Define parameters
parameters = {
    # 'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000),
    'tfidf__use_idf': (True, False),
    # 'clf__estimator__n_estimators': [50, 100, 200],
    'clf__estimator__min_samples_split': [2, 3, 4]
}
```

As it can be seen, some of the possible parameters and their ranges are ultimately commented out in the code: this is because training a multi-output classifier like this on the full list proved out to be _extremely_ time-consuming. After a few tries I decided to settle for the uncommented ones: I cannot say that I have seen a marked improvement in performance overall, but it does illustrate the possibility to optimize every step of ML pipeline, given thet we manipulate parameters for the transformers as well as the classifier.  
Training of this configuration required slightly more than 8 hours on an AWS-hosted, ML-specific VM. 

## License
 <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
