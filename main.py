from transformers import pipeline, set_seed
from datasets import load_dataset

def main():
    dataset = load_dataset("aakanksha/udpos")
    print(dataset["train"][0])
    print(dataset["test"][0])

if __name__ == '__main__':
    main()