import zipfile
import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shutil

def combine_csvs(csvs, output_filename, airport):
        
    csvs = sorted(csvs)
    
    # Create parquet writer with schema from first file
    first_chunk = next(pd.read_csv(csvs[0], dtype=str, chunksize=1))
    first_chunk = first_chunk.fillna('')
    schema = pa.Table.from_pandas(first_chunk).schema
    
    writer = pq.ParquetWriter(output_filename, schema)
    
    for i, file in enumerate(csvs):
        total_csvs = len(csvs)
        print(f"\rProcessing raw data \t airport: {airport} \t {i+1}/{total_csvs}", end='', flush=True)
        for chunk in pd.read_csv(file, dtype=str, chunksize=10000):
            chunk = chunk.fillna('')
            table = pa.Table.from_pandas(chunk)
            writer.write_table(table)
    
    writer.close()


def process_airport(airport, mode, temp_dir='processed_data/temp'):
    tfm_pattern = f'{temp_dir}/**/{airport}_????-??-??.TFM_track_data_set.csv'
    tbfm_pattern = f'{temp_dir}/**/{airport}_????-??-??.TBFM_data_set.csv'
    runways_pattern = f'{temp_dir}/**/{airport}_????-??-??.runways_data_set.csv'
    os.makedirs(f'processed_data/{mode}', exist_ok=True)
    combine_csvs(glob.glob(tfm_pattern, recursive=True), f'processed_data/{mode}/{airport}_tfm.parquet', airport)
    combine_csvs(glob.glob(tbfm_pattern, recursive=True), f'processed_data/{mode}/{airport}_tbfm.parquet', airport)
    combine_csvs(glob.glob(runways_pattern, recursive=True), f'processed_data/{mode}/{airport}_runways.parquet', airport)
    

def unzip_folder(zip_path, temp_dir='processed_data/temp'):
    os.makedirs(temp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)


def clean_up(temp_dir='processed_data/temp'):
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    airports = ['KATL', 'KCLT', 'KDEN', 'KDFW', 'KJFK', 'KMEM', 'KMIA', 'KORD', 'KPHX', 'KSEA']
    os.makedirs('processed_data/train', exist_ok=True)
    os.makedirs('processed_data/test', exist_ok=True)

    # Process test data
    test_zip = 'raw_data/test/FUSER_test.zip'
    unzip_folder('raw_data/test/FUSER_test.zip')
    for airport in airports:
        process_airport(airport, 'test')
    clean_up()
    # os.remove(test_zip)

    # Process train data
    for train_zip in os.listdir('raw_data/train'):
        unzip_folder(f'raw_data/train/{train_zip}')
        airport = train_zip.split('_')[-1].replace('.zip', '')
        process_airport(airport, 'train')
        clean_up()
        # os.remove(train_zip)