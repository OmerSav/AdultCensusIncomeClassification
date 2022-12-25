Adult Census Income Classification
==============================
The project task
from "[Adult Census Income -
Predict whether income exceeds $50K/yr based on census data](https://www.kaggle.com/datasets/uciml/adult-census-income)"

### Data Description from Kaggle

> This data was extracted from the 1994 Census bureau database by Ronny Kohavi
> and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of
> reasonably clean records was extracted using the following conditions:
> ((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)). The prediction task is
> to determine whether a person makes over $50K a year.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── predictions    <- After using model predictions saved here.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │ │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis
    │                          environment.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <-  Scripts to generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

# **Adult Census Income Classification**

I am working on this task for practicing and demonstrating my expertise in Data
Science.
My main purpose for this project is to show my data science skills without
revealing my client's projects.
I didn't spend time to make this project perfect, and because of that, there can
be mistakes. It is for demonstration purposes without ethical problems.

## Project Structure Explanation

I mainly worked and made every operation on jupyter notebooks to show and
explain ecery step I did,
So you can basically check **notebooks** folder to see them also you can find
pdf versions of these notebooks with same name in
**reports** folder.
Also you can find helper function on **src/utils.py** and data preparation
training, prediction
codes in **src**. This scripts coded for using easily on command line with
*make*
commmads (you can see detailed command prompt api from below).

## Model Training and Performance

### Models

I tried every model as hand-tuned or with grid search I reported all results.
You can see models and frameworks/libraries in the table below.

| Model                     | Library - Framework |
|:--------------------------|:-------------------:|
| Logistic Regression       |    scikit-learn     |
| Support Vector Machine    |    scikit-learn     |
| K-Nearest Neighbors       |    scikit-learn     |
| Decision Tree             |    scikit-learn     |
| Random Forest             |    scikit-learn     |
| Ada Boost                 |    scikit-learn     |
| Gradient Boosting         |    scikit-learn     |
| eXtreme Gradient Boosting |    XGBoost          |
| Neural Network            |     TensorFlow      |

### Performance on Test Set

I demonstrated the scores from the test set below table. The test set didn't use
when training the model and choosing parameters for validation purposes. The
test set included after performance evaluation and trained final models.
To validate, k-fold cross-validation was used on the training set.

If you would like to you can also see the training set performance from
**reports/0.2-os-train-models.pdf**.

##### Test Set Macro Avg. Performances

<table class="dataframe" style="border:1; border-color: grey">
<thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Logistic Regression</th>
      <td>0.7562</td>
      <td>0.8226</td>
      <td>0.7741</td>
      <td>0.8145</td>
    </tr>
    <tr>
      <th>K-Nearest Neighbors</th>
      <td>0.7206</td>
      <td>0.7909</td>
      <td>0.7314</td>
      <td>0.7709</td>
    </tr>
    <tr>
      <th>Support Vector Machine</th>
      <td>0.7418</td>
      <td>0.8204</td>
      <td>0.7542</td>
      <td>0.7895</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.6970</td>
      <td>0.7739</td>
      <td>0.6857</td>
      <td>0.7140</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.7447</td>
      <td>0.8133</td>
      <td>0.7608</td>
      <td>0.8010</td>
    </tr>
    <tr>
      <th>Ada Boost</th>
      <td>0.7688</td>
      <td>0.8384</td>
      <td>0.7879</td>
      <td>0.8259</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.7729</td>
      <td>0.8412</td>
      <td>0.7925</td>
      <td>0.8305</td>
    </tr>
    <tr>
      <th>eXtreme Gradient Boosting</th>
      <td>0.7712</td>
      <td>0.8411</td>
      <td>0.7906</td>
      <td>0.8282</td>
    </tr>
    <tr>
      <th>Neural Network</th>
      <td>0.7435</td>
      <td>0.8087</td>
      <td>0.7600</td>
      <td>0.8019</td>
    </tr>
  </tbody>
</table>


## Custom Data Preparation & Training & Prediction

### *make* data

It generates handled and encoded data from raw data for training. There is
already **adult.csv** data to convert but if you want to train on different data
with the same structure and column names, simply put data in **data/raw/**
folder with the naming **adult.csv** and use this command from the project
content root.

### *make* train_model

**Prerequests**

Proccessed training data.(*make data* command)

**Usage**

It trains model with **data/proccesed/adult.csv** and then saves trained models
to **models/trained/** folder.
It also saves standart scaler and polynomial converter
to **models/featurebuild/**.

### *make* predict

**Prerequests**

Trained models. It is already exist but if there is no model saved you can
simply use *make train_model* command.

**Usage**

It takes two arguments *model*, *data*

> make predict model='gradient-boosting' data='.../somedatafolder/somedata.csv'

Model argument could be any of them below:

```
'logistic', 'K-NN', 'SVM', 'decision-tree', 'random-forest',
'ada-boost', 'gradient-boosting', 'XGBR', 'neural-network'
```

The data argument could be any file path with data that has the same structure
as **data/raw/test.csv**.

It predicts data and saves the result as CSV to **data/predictions/** folder
with the name of the original file and time stamp following it.
