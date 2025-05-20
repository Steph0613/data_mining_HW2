import os
import pandas as pd
from pyarrow.parquet import ParquetFile
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import MultiLabelBinarizer

INPUT_DIR = './outputs/expanded_items_chunks'
OUTPUT_DIR = './outputs/task4'
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRANS_CSV = os.path.join(OUTPUT_DIR, 'task4_refund_transactions.csv')
RULES_CSV = os.path.join(OUTPUT_DIR, 'task4_rules.csv')
FREQ_CSV = os.path.join(OUTPUT_DIR, 'task4_frequent_itemsets.csv')
MIN_SUPPORT = 0.005
MIN_CONFIDENCE = 0.4
CHUNKSIZE = 500_000

transactions = []

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.endswith('.parquet'):
        continue
    pf = ParquetFile(os.path.join(INPUT_DIR, fname))
    for batch in pf.iter_batches(batch_size=CHUNKSIZE):
        df = batch.to_pandas()
        df = df.dropna(subset=['payment_status', 'item_category'])
        df = df[df['payment_status'].isin(['已退款', '部分退款'])]
        grouped = df.groupby(['id', 'purchase_date'])
        for _, group in grouped:
            refund_status = group['payment_status'].iloc[0]
            label = 'STATUS_已退款' if refund_status == '已退款' else 'STATUS_部分退款'
            cats = list(set(group['item_category']))
            if cats:
                transactions.append([label] + cats)

pd.Series(transactions).to_csv(TRANS_CSV, index=False, header=False)

mlb = MultiLabelBinarizer()
onehot = pd.DataFrame(mlb.fit_transform(transactions), columns=mlb.classes_)
freq_itemsets = fpgrowth(onehot, min_support=MIN_SUPPORT, use_colnames=True)
rules = association_rules(freq_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)

freq_itemsets.to_csv(FREQ_CSV, index=False)
rules.to_csv(RULES_CSV, index=False)