import argparse
import os
parser = argparse.ArgumentParser()
pa = parser.add_argument
pa("--datapath", type=str, default="data")
pa("--batch_size", type=int, default="128")
pa("--lstm_size", type=int, default="1024")
pa("--max_sequence", type=int, default="400")
pa("--frange", type=int, default="0")
pa("--execution_gpu", type=int, default='0')

args = parser.parse_args()

path_dict = {
    "training_mnli": os.path.join(args.datapath, "multinli_0.9","multinli_0.9_train.jsonl"),
    "dev_matched": os.path.join(args.datapath, "multinli_0.9","multinli_0.9_dev_matched.jsonl"),
    "dev_mismatched": os.path.join(args.datapath, "multinli_0.9","multinli_0.9_dev_mismatched.jsonl"),
    "test_matched": os.path.join(args.datapath, "multinli_0.9","multinli_0.9_test_matched_unlabeled.jsonl"),
    "test_mismatched": os.path.join(args.datapath, "multinli_0.9", "multinli_0.9_test_mismatched_unlabeled.jsonl"),
    "training_snli": os.path.join(args.datapath, "snli_1.0", "snli_1.0_train.jsonl"),
    "dev_snli": os.path.join(args.datapath,"snli_1.0", "snli_1.0_dev.jsonl"),
    "test_snli": os.path.join(args.datapath, "snli_1.0","snli_1.0_test.jsonl"),
    "embedding_data_path": os.path.join(args.datapath,"glove.840B.300d.txt"),
}
