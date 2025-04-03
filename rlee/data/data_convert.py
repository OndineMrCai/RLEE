import json

import argparse
parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
parser.add_argument('--datapath', default="/home/sccai/projects/def-hongyanz/sccai/dataset/orz/orz_math_57k_collected.json",
                   help='Local directory to save processed datasets')
parser.add_argument('--savepath', default="./train/orz_math_57k_collected.json")
args = parser.parse_args()
datapath = args.datapath
data = json.load(open(datapath))


newdata = [{'question': d[0]['value'], 'answer': d[1]['ground_truth']['value']} for d in data]
#save to a new json file
savepath=args.savepath
json.dump(newdata, open(savepath, 'w'))