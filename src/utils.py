from IPython.display import HTML, display
import numpy as np
import pandas as pd


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

    html_table(count_unique)
    html_table(categories)
