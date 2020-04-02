from bpemb import BPEmb
import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--output-dir", type=str, default="data/embeddings")
    parser.add_argument("--vs", type=int, default=200000)
    args = parser.parse_args()

    bpe = BPEmb(lang=args.lang, vs=args.vs)

    with open(
        os.path.join(args.output_dir, 
        "bpe_{}_{}.txt".format(args.lang, args.vs)), "w") as f:
        for i in tqdm(range(bpe.vectors.shape[0])):
            w = bpe.decode_ids([i])
            w = w.replace(" ", "")
            vec = bpe.vectors[i]
            f.write(w + " " + " ".join([str(vec[j]) for j in range(bpe.vectors.shape[1])]) + "\n")
