import os
import json
import time
import pandas as pd
from pandas import json_normalize
from pyarrow.parquet import ParquetFile
from pyarrow import Table, parquet as pq
from hashlib import sha256
from pybloom_live import ScalableBloomFilter as BaseSBF
from collections import defaultdict
from tqdm import tqdm  

INPUT_DIR = './data'
OUTPUT_DIR = './outputs/cleaned_chunks'
LOG_FILE = './outputs/cleaning_log.txt'
CHUNKSIZE = 5_000_000
os.makedirs(OUTPUT_DIR, exist_ok=True)

def column_loader(parquet_dir, columns=None, chunksize=CHUNKSIZE):
    expanded_fields = {'purchase_avg_price', 'purchase_categories'}
    files = [file for file in os.listdir(parquet_dir) if file.endswith('.parquet')]
    for file in tqdm(files, desc='处理文件列表'):
        file_path = os.path.join(parquet_dir, file)
        print(f"正在读取文件: {file}")
        pf = ParquetFile(file_path)
        has_raw = 'purchase_history' in pf.schema.names
        if has_raw:
            actual_columns = [col for col in columns if col not in expanded_fields] if columns else pf.schema.names
            if any(col in expanded_fields for col in columns or []):
                actual_columns.append('purchase_history')
            process_mode = 'raw'
        else:
            actual_columns = columns
            process_mode = 'clean'
        for batch in pf.iter_batches(columns=actual_columns, batch_size=chunksize):
            df = batch.to_pandas()
            if process_mode == 'raw' and 'purchase_history' in df.columns:
                df = process_expansion(df, columns)
            df = apply_dtype_optimization(df)
            yield df

def process_expansion(df, required_fields=None):
    try:
        parsed = df['purchase_history'].apply(lambda x: json.loads(x) if pd.notnull(x) else {})
        df_purchase = json_normalize(parsed).rename(columns={
            'avg_price': 'purchase_avg_price',
            'categories': 'purchase_categories',
            'items': 'purchase_item_ids',
            'payment_method': 'payment_method',
            'payment_status': 'payment_status',
            'purchase_date': 'purchase_date'
        })
        merged = pd.concat([
            df.drop('purchase_history', axis=1),
            df_purchase[[col for col in df_purchase.columns if required_fields is None or col in required_fields]]
        ], axis=1)
        return merged
    except Exception as e:
        print(f"字段展开失败: {str(e)}")
        return df.drop('purchase_history', axis=1, errors='ignore')

def apply_dtype_optimization(df):
    dtype_map = {
        'id': 'uint32',
        'age': 'uint8',
        'income': 'float32',
        'gender': 'category',
        'country': 'category',
        'is_active': 'bool',
        'purchase_avg_price': 'float32',
        'purchase_categories': 'category',
        'payment_method': 'category',
        'payment_status': 'category',
        'item_category': 'category'
    }
    valid = {col: dtype for col, dtype in dtype_map.items() if col in df.columns}
    return df.astype(valid, errors='ignore')

class ScalableBloomFilter:
    def __init__(self, error_rate=0.001):
        self.filter = BaseSBF(error_rate=error_rate)
    def add(self, item):
        exists = item in self.filter
        self.filter.add(item)
        return exists

def save_clean_chunk(df, output_dir, chunk_idx):
    path = os.path.join(output_dir, f"clean_{chunk_idx}.parquet")
    table = Table.from_pandas(df)
    pq.write_table(table, path, compression='brotli', use_dictionary=False)

def preprocess(parquet_dir, output_dir, columns_to_keep=None):
    if columns_to_keep is None:
        columns_to_keep = [
            'id', 'last_login', 'user_name', 'fullname', 'age', 'income',
            'gender', 'country', 'is_active',
            'purchase_avg_price', 'purchase_categories',
            'purchase_item_ids', 'payment_method', 'payment_status', 'purchase_date'
        ]
    NUMERIC_COLS = ['age', 'income', 'purchase_avg_price']
    dedup_filter = ScalableBloomFilter(error_rate=0.0001)
    stats = {
        'duplicates_removed': 0,
        'missing_filled': defaultdict(int),
        'outliers_removed': defaultdict(int),
        'chunks_processed': 0
    }
    chunk_gen = column_loader(parquet_dir, columns=columns_to_keep)
    for idx, chunk in enumerate(chunk_gen):
        print(f"正在处理数据块 {idx}...")
        chunk = chunk[[col for col in columns_to_keep if col in chunk.columns]]
        key_hash = (chunk['id'].astype(str) + '_' + chunk['last_login'].astype(str) + '_' + chunk['user_name'].astype(str)).apply(lambda x: sha256(x.encode()).hexdigest())
        mask = key_hash.apply(lambda x: not dedup_filter.add(x))
        stats['duplicates_removed'] += (~mask).sum()
        chunk = chunk[mask]
        before = len(chunk)
        chunk = chunk.dropna(subset=['id', 'last_login', 'user_name', 'age', 'income'])
        stats['missing_filled']['total'] += before - len(chunk)
        for col in NUMERIC_COLS:
            if col in chunk.columns:
                q1 = chunk[col].quantile(0.25)
                q3 = chunk[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (chunk[col] >= lower) & (chunk[col] <= upper)
                stats['outliers_removed'][col] += (~mask).sum()
                chunk = chunk[mask]
        save_clean_chunk(chunk, output_dir, idx)
        stats['chunks_processed'] += 1
    return stats

if __name__ == '__main__':
    start_time = time.time()
    stats = preprocess(INPUT_DIR, OUTPUT_DIR)
    elapsed = time.time() - start_time
    print("清洗完成！")
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"运行时间: {elapsed:.2f} 秒\n")
        f.write(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    print(f"总耗时: {elapsed:.2f} 秒，日志已保存至 {LOG_FILE}")
