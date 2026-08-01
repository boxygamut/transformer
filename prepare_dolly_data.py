"""Convert databricks-dolly-15k.jsonl into the flat <human>/<bot> text
format already used by Dataset.txt, so it can be fed through the same
load_dataset() / tokenizer.train() pipeline in transformer.ipynb.
"""
import argparse
import json
import random

EOS = "<endOfText>"


def format_example(example: dict) -> str:
    instruction = example["instruction"].strip()
    context = example.get("context", "").strip()
    response = example["response"].strip()

    prompt = f"{instruction}\n\n{context}" if context else instruction

    return f"<human> {prompt}{EOS}\n<bot> {response}{EOS}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="databricks-dolly-15k.jsonl")
    parser.add_argument("--output", default="dolly_formatted.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    # Dolly is loosely grouped by category; shuffle so the notebook's
    # contiguous 90/10 train/test split isn't biased toward whatever
    # category happens to land at the end of the file.
    random.Random(args.seed).shuffle(examples)

    with open(args.output, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(format_example(example))

    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
