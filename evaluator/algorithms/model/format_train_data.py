import argparse
import json
import random
import torch
from e5 import E5Model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="intfloat/e5-base-v2")
parser.add_argument("--train_data_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--num_hard_negatives", type=int, default=5)
parser.add_argument("--num_positives", type=int, default=5)
args = parser.parse_args()

model = E5Model(args.model)

with open(args.train_data_path, "r") as f:
    train_data = json.load(f)

# Build a mapping from tool_name to all queries for that tool
from collections import defaultdict

tool_to_queries = defaultdict(list)
for entry in train_data:
    tool_to_queries[entry["tool_name"]].append(entry["query"])

all_tool_names = list(tool_to_queries.keys())

all_triplets = []
for entry in train_data:
    anchor_query = entry["query"]
    anchor_tool = entry["tool_name"]
    # Positives: other queries for the same tool (excluding the anchor)
    positives = [q for q in tool_to_queries[anchor_tool] if q != anchor_query]
    if not positives:
        continue  # skip if no positive
    positives = positives[:args.num_positives]
    # Hard negatives: sample queries from other tools
    negative_tools = [t for t in all_tool_names if t != anchor_tool]
    hard_negatives = []
    for t in random.sample(negative_tools, min(len(negative_tools), args.num_hard_negatives * 2)):
        hard_negatives.extend(tool_to_queries[t])
        if len(hard_negatives) >= args.num_hard_negatives:
            break
    hard_negatives = hard_negatives[:args.num_hard_negatives]
    # Form triplets
    triplets = [
        (anchor_query, positive, negative)
        for positive in positives
        for negative in hard_negatives
    ]
    all_triplets.extend(triplets)

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets):
        self.triplets = triplets
    def __len__(self):
        return len(self.triplets)
    def __getitem__(self, idx):
        return self.triplets[idx]

dataset = TripletDataset(all_triplets)
torch.save(dataset, args.output_path)
