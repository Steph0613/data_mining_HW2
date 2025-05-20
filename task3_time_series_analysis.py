import os
import pandas as pd
from pyarrow.parquet import ParquetFile
from collections import defaultdict, Counter

INPUT_DIR = './outputs/expanded_items_chunks'
OUTPUT_DIR = './outputs/task3'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHUNKSIZE = 500_000

quarter_count = defaultdict(Counter)
weekday_count = defaultdict(Counter)
sequence_count = defaultdict(int)

last_user_date_category = {}

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.endswith('.parquet'):
        continue
    pf = ParquetFile(os.path.join(INPUT_DIR, fname))
    for batch in pf.iter_batches(batch_size=CHUNKSIZE):
        df = batch.to_pandas()
        df = df.dropna(subset=['id', 'item_category', 'purchase_date'])
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['quarter'] = df['purchase_date'].dt.quarter.astype(str)
        df['weekday'] = df['purchase_date'].dt.weekday.astype(str)
        for row in df.itertuples():
            quarter_count[row.item_category][row.quarter] += 1
            weekday_count[row.item_category][row.weekday] += 1
            uid = row.id
            date = row.purchase_date
            key = (uid,)
            last = last_user_date_category.get(key)
            if last and last[0] < date:
                sequence = (last[1], row.item_category)
                sequence_count[sequence] += 1
            last_user_date_category[key] = (date, row.item_category)

quarter_df = pd.DataFrame(quarter_count).fillna(0).astype(int).T
quarter_df.columns = [str(c) for c in quarter_df.columns]
quarter_df.to_csv(os.path.join(OUTPUT_DIR, 'task3_quarterly_category_counts.csv'))

weekday_df = pd.DataFrame(weekday_count).fillna(0).astype(int).T
weekday_df.columns = [str(c) for c in weekday_df.columns]
weekday_df.to_csv(os.path.join(OUTPUT_DIR, 'task3_weekday_category_counts.csv'))

sequence_df = pd.DataFrame([{'from_category': k[0], 'to_category': k[1], 'count': v} for k, v in sequence_count.items()])
sequence_df.sort_values(by='count', ascending=False, inplace=True)
sequence_df.to_csv(os.path.join(OUTPUT_DIR, 'task3_sequential_category_pairs.csv'), index=False)