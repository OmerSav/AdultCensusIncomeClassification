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

##### Test Set Performances

# Tablo degisecek

<table class="dataframe" style="border:1; border-color: grey">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R2</th>
      <th>MAE</th>
      <th>MSE</th>
      <th>RMSE</th>
      <th>RMSLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Multiple Regression</th>
      <td>0.851</td>
      <td>16853.092</td>
      <td>696571233.702</td>
      <td>26392.635</td>
      <td>0.157</td>
    </tr>
    <tr>
      <th>Polinomial Regression</th>
      <td>0.546</td>
      <td>24949.286</td>
      <td>2564744078.439</td>
      <td>50643.302</td>
      <td>0.217</td>
    </tr>
    <tr>
      <th>Lasso Regression (l1 regularization)</th>
      <td>0.886</td>
      <td>14843.647</td>
      <td>529524008.165</td>
      <td>23011.388</td>
      <td>0.132</td>
    </tr>
    <tr>
      <th>Ridge Regression (l2 regularization)</th>
      <td>0.887</td>
      <td>14888.542</td>
      <td>526874799.133</td>
      <td>22953.753</td>
      <td>0.133</td>
    </tr>
    <tr>
      <th>ElasticNet Regression (l1 and l2 regularization)</th>
      <td>0.883</td>
      <td>15103.082</td>
      <td>544291083.688</td>
      <td>23330.046</td>
      <td>0.132</td>
    </tr>
    <tr>
      <th>Support Vector Regression</th>
      <td>0.293</td>
      <td>39896.220</td>
      <td>3308174465.590</td>
      <td>57516.732</td>
      <td>0.306</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.717</td>
      <td>24569.233</td>
      <td>1321987229.902</td>
      <td>36359.142</td>
      <td>0.198</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.864</td>
      <td>16378.079</td>
      <td>635300656.327</td>
      <td>25205.171</td>
      <td>0.147</td>
    </tr>
    <tr>
      <th>Ada Boost</th>
      <td>0.810</td>
      <td>21721.918</td>
      <td>886924139.055</td>
      <td>29781.271</td>
      <td>0.192</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.878</td>
      <td>14810.326</td>
      <td>569118586.082</td>
      <td>23856.206</td>
      <td>0.133</td>
    </tr>
    <tr>
      <th>eXtreme Gradient Boosting</th>
      <td>0.866</td>
      <td>15624.645</td>
      <td>623024752.327</td>
      <td>24960.463</td>
      <td>0.138</td>
    </tr>
    <tr>
      <th>Neural Network</th>
      <td>0.874</td>
      <td>16079.370</td>
      <td>585419850.640</td>
      <td>24195.451</td>
      <td>0.144</td>
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
