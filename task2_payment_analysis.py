import os
import pandas as pd
from pyarrow.parquet import ParquetFile
from collections import defaultdict
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import MultiLabelBinarizer

INPUT_DIR = './outputs/expanded_items_chunks'
TASK2_OUTPUT_DIR = './outputs/task2'
os.makedirs(TASK2_OUTPUT_DIR, exist_ok=True)
CATEGORY_TRANS_CSV = os.path.join(TASK2_OUTPUT_DIR, 'task2_payment_category_transactions.csv')
RULES_OUTPUT_CSV = os.path.join(TASK2_OUTPUT_DIR, 'task2_payment_category_rules.csv')
HIGH_VALUE_STATS_CSV = os.path.join(TASK2_OUTPUT_DIR, 'task2_high_value_by_payment.csv')
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.6
CHUNKSIZE = 500_000

payment_category_transactions = []
high_value_counts = defaultdict(int)
high_value_total = 0

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.endswith('.parquet'):
        continue
    pf = ParquetFile(os.path.join(INPUT_DIR, fname))
    for batch in pf.iter_batches(batch_size=CHUNKSIZE):
        df = batch.to_pandas()
        df = df.dropna(subset=['item_category', 'payment_method', 'purchase_date'])
        grouped = df.groupby(['payment_method', 'purchase_date'])
        for (payment, date), group in grouped:
            cats = set(group['item_category'])
            if cats:
                payment_category_transactions.append([payment] + list(cats))
        high_value = df[df['is_high_value'] == True]
        for method, count in high_value['payment_method'].value_counts().items():
            high_value_counts[method] += count
            high_value_total += count

pd.Series(payment_category_transactions).to_csv(CATEGORY_TRANS_CSV, index=False, header=False)

mlb = MultiLabelBinarizer()
onehot = pd.DataFrame(mlb.fit_transform(payment_category_transactions), columns=mlb.classes_)
freq_itemsets = fpgrowth(onehot, min_support=MIN_SUPPORT, use_colnames=True)
rules = association_rules(freq_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules.to_csv(RULES_OUTPUT_CSV, index=False)

high_value_df = pd.DataFrame(list(high_value_counts.items()), columns=['payment_method', 'count'])
high_value_df['ratio'] = high_value_df['count'] / high_value_total
high_value_df.to_csv(HIGH_VALUE_STATS_CSV, index=False)