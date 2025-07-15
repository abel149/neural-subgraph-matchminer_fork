DATASET=graph.pkl
COUNTS=results/

.PHONY: all matcher miner counter analyze

all: matcher miner counter analyze

matcher:
	python -m subgraph_matching.train --node_anchored

miner:
	python -m subgraph_mining.decoder --dataset=$(DATASET) --node_anchored

counter:
	python -m analyze.count_patterns --dataset=$(DATASET) --out_path=results/counts.json --node_anchored --preserve_labels

analyze:
	python -m analyze.analyze_pattern_counts --counts_path=$(COUNTS)
