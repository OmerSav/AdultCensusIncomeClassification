import os
from pathlib import Path
import logging
from joblib import dump
import json

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
import xgboost as xgb
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from src.utils import find_threshold


def main():
    logger = logging.getLogger(__name__)
    logger.info('Training models...')
    processed_data_dir = Path('./data/processed/')
    file_name = 'adult.csv'
    file_path = processed_data_dir / file_name
    models_trained_dir = Path('./models/trained/')
    if not os.path.exists(models_trained_dir):
        os.mkdir(models_trained_dir)
    featurebuild_dir = Path('./models/featurebuild/')
    if not os.path.exists(featurebuild_dir):
        os.mkdir(featurebuild_dir)
    threshold_file = featurebuild_dir / '0.3-besttreshold.json'
    if not os.path.exists(threshold_file):
        json.dump({}, open(threshold_file, 'wt'))

    df = pd.read_csv(file_path)
    X = df.drop('income', axis=1)
    y = df['income']

    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    def save_threshold(model, model_name, tf_model=False):
        y_vals = y
        if (isinstance(y_vals, pd.Series)):
            y_vals = y_vals.values
        if tf_model:
            y_pred_proba = model.predict(X)[:, 0]
        else:
            y_pred_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_vals, y_pred_proba)
        roc_vals = pd.DataFrame(
            {'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
        best_treshold_ = find_threshold(roc_vals)
        thresholds_ = json.load(open(threshold_file))
        thresholds_[model_name] = float(best_treshold_)
        json.dump(thresholds_, open(threshold_file, 'wt'))

    def train_deploy(model_class, X, model_name, **kwargs):
        model = model_class(**kwargs)
        model.fit(X, y)
        save_threshold(model, model_name)
        model_path = models_trained_dir / f'{model_name}.joblib'
        dump(model, model_path)
        return model

    log_final = train_deploy(LogisticRegression, X_scaled,
                             '0.1-logisticregression',
                             **{'C': 0.1, 'l1_ratio': 0.0, 'penalty': 'l1',
                                'solver': 'saga'})
    knn_final = train_deploy(KNeighborsClassifier, X_scaled,
                             '0.2-knearestneighbors',
                             **{'n_neighbors': 15})
    svc_final = train_deploy(SVC, X_scaled, '0.3-supportvector',
                             **{'C': 1, 'kernel': 'linear',
                                'probability': True})
    tree_final = train_deploy(DecisionTreeClassifier, X, '0.4-decisiontree',
                              **{'criterion': 'gini', 'max_depth': 5,
                                 'min_impurity_decrease': 0.01,
                                 'min_samples_split': 3})
    rfc_final = train_deploy(RandomForestClassifier, X, '0.5-randomforest',
                             **{'criterion': 'gini', 'n_estimators': 200})
    ada_final = train_deploy(AdaBoostClassifier, X, '0.6-adaboost',
                             **{'learning_rate': 1.0, 'n_estimators': 1000})
    gbc_final = train_deploy(GradientBoostingClassifier, X,
                             '0.7-gradientboosting',
                             **{'max_depth': 5, 'n_estimators': 100})
    xgbc_final = train_deploy(xgb.XGBClassifier, X,
                              '0.8-extremegradientboosting',
                              **{'max_depth': 5, 'n_estimators': 100})

    # neural network 

    def ann_model_init():
        inputs = tf.keras.Input(shape=(X_scaled.shape[1]))
        x = tf.keras.layers.Dense(10, 'relu')(inputs)
        x = tf.keras.layers.Dense(20, 'relu')(x)
        x = tf.keras.layers.Dense(10, 'relu')(x)

        outputs = tf.keras.layers.Dense(1, 'sigmoid')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    ann_model = ann_model_init()

    @tf.function
    def weighted_binary_crossentropy(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        epsilon_ = 1e-7
        return -tf.reduce_sum(
            2. * y_true * tf.math.log(y_pred + epsilon_) + (
                    1. - y_true) * tf.math.log(
                1. - y_pred + epsilon_), axis=-1)

    optimizer = tf.keras.optimizers.Adam(0.005)
    metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    compile_params = {'optimizer': optimizer,
                      'loss': weighted_binary_crossentropy,
                      'metrics': [metric]}

    ann_model.compile(**compile_params)

    ann_model.fit(x=X_scaled, y=y,
                  epochs=500,
                  batch_size=X_scaled.shape[0],
                  verbose=0,
                  shuffle=True)
    ann_model.save(models_trained_dir / f'0.9-neuralnet.h5')
    save_threshold(ann_model, '0.9-neuralnet.h5', tf_model=True)

    # save scaler
    dump(final_scaler, featurebuild_dir / '0.2-standardscaler.joblib')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
