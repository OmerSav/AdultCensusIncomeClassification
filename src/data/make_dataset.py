# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
import joblib

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from src.utils import build_features


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    inp_dir = Path(input_filepath)
    data_files = os.listdir(input_filepath)
    enc = joblib.load('./models/featurebuild/0.1-onehotencoder.joblib')
    for file in data_files:
        df = pd.read_csv(inp_dir / file)
        df_final = build_features(df, enc)
        df_final.to_csv(Path(output_filepath) / file, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
