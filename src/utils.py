from IPython.display import HTML, display
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, \
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, roc_curve
from sklearn.preprocessing import OneHotEncoder


def html_table(df):
    if (isinstance(df, pd.Series)):
        df = df.to_frame()
    display(HTML(df.to_html()))


def categories_show(categorical_df):
    count_unique = categorical_df.nunique().sort_values(ascending=False)
    max_cat = count_unique.max()
    categories = pd.DataFrame(columns=range(max_cat))
    for col_name in count_unique.index:
        unqs = categorical_df[col_name].unique()
        if len(unqs) == max_cat:
            categories.loc[col_name] = unqs
            continue
        categories.loc[col_name] = np.concatenate(
            (unqs, np.array([np.nan, ] * (max_cat - len(unqs)))))
    count_unique.columns = ['Unique Value Count']
    html_table(count_unique)
    html_table(categories)


def classification_scores(model, X, y, threshold=0.5, plots=True,
                          tf_model=False):
    if (isinstance(y, pd.Series)):
        y = y.values
    if tf_model:
        y_pred_proba = model.predict(X)[:, 0]
    else:
        y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype('int32')
    cls_reports = classification_report(y, y_pred, output_dict=True)
    confusion = pd.DataFrame(confusion_matrix(y, y_pred), index=[0, 1],
                             columns=['Predicted 0', 'Predicted 1'])
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    roc_vals = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})

    print(classification_report(y, y_pred))
    ConfusionMatrixDisplay.from_predictions(y, y_pred)
    if plots:
        RocCurveDisplay.from_predictions(y, y_pred_proba)
        PrecisionRecallDisplay.from_predictions(y, y_pred_proba)

    scores = {'classification_reports': cls_reports,
              'confusion_matrix': confusion, 'roc_curve': roc_vals}
    return scores


def find_threshold(roc_curve_df):
    '''Find best threshold for our task.
    We put equal importance on predicting class so, we get an average of
    (1-fpr)+tpr and find a threshold to maximize that value.
    '''
    df = roc_curve_df.copy()
    df['mean'] = (1 - df['FPR'] + df['TPR']) / 2
    ind = df['mean'].argmax()
    threshold = df['Thresholds'][ind]
    print(f'\n\t The best threshold for balancing recall is: {threshold}\n')
    return threshold


def build_features(df: pd.DataFrame, enc: OneHotEncoder) -> pd.DataFrame:
    '''Process new data to enter model without scaling
    Operations -> Handle missings, apply one hot encoding on data.

    Args:
        df (pandas.DataFrame): df to procces
        enc (sklearn.preprocessing.OneHotEncoder): One hot encoder fitted on
        train set

    Returns:
        (pd.DataFrame): Dataset for models without scaling
    '''
    print(f'Before drop total row is: {len(df)}')
    if 'SalePrice' in df.columns:
        df = df[df['SalePrice'] < 500000]
    df.dropna(axis=0, subset=['Electrical', 'MasVnrArea'], inplace=True)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())
    df = df.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1)
    df['Fence'] = df['Fence'].fillna('None')
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda value: value.fillna(value.mean()))
    df.drop(axis=1, columns=['Id'], inplace=True)
    nullable_cols = ['GarageFinish', 'GarageQual', 'GarageCond', 'GarageType',
                     'BsmtCond',
                     'BsmtExposure', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']
    fill_cols = [col for col in df.columns if col not in nullable_cols]
    df.dropna(axis=0, subset=fill_cols, inplace=True)
    print(f'After drop total row is: {len(df)}')

    df.reset_index(inplace=True, drop=True)
    df['MSSubClass'] = df['MSSubClass'].apply(str)
    object_df = df.select_dtypes(include='object')
    numeric_df = df.select_dtypes(exclude='object')
    df_objects_dummies = enc.transform(object_df)
    df_encoded = pd.concat((numeric_df, pd.DataFrame(df_objects_dummies)),
                           axis=1)
    print(df_encoded.shape)
    return df_encoded
