![build](https://github.com/JlKmn/ProbingPretrainedLM/actions/workflows/ci.yml/badge.svg)
# Probing Pre-trained Language Models

## Getting started
Assuming Linux / Git Bash
1. Clone the repo\
`git clone https://github.com/JlKmn/ProbingPretrainedLM.git && cd ProbingPretrainedLM`
1. Create virtualenv\
`python -m venv env`
2. Activate virtualenv\
`source env/bin/activate`
3. Install requirements\
`pip install -r requirements.txt`
4. Log into wandb\
`wandb login`
5. Start training\
`python run.py --model 1 --dataset pos --epochs 5`

## Results
### POS
![accuracy](https://raw.githubusercontent.com/JlKmn/ProbingPretrainedLM/main/results/pos_accuracy.png)
### NER
![recall](https://raw.githubusercontent.com/JlKmn/ProbingPretrainedLM/main/results/ner_recall.png)
![precision](https://raw.githubusercontent.com/JlKmn/ProbingPretrainedLM/main/results/ner_precision.png)