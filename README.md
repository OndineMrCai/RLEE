# RLEE

## Setup


## Data
Convert training data and test data into parquet format
```
python RLEE/rlee/data/data_convert.py \
 --datapath /home/sccai/projects/def-hongyanz/sccai/dataset/orz/orz_math_57k_collected.json \
 --savepath /home/sccai/projects/def-hongyanz/sccai/RLEE/rlee/data/train/orz_math_57k_collected.json

python RLEE/rlee/data/rlee_dataset.py \
 --local_dir /home/sccai/projects/def-hongyanz/sccai/RLEE/rlee/data/train/
```

## Acknowledge