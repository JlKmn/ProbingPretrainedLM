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
Training Hyperparameters 
| Epochs  | Batch size | Loss rate |
| ------------- | ------------- | ------------- |
| 50  | 64  | 0.01 |

The LinearBert model is an exception and was trained with an initial loss rate of 0.0001

### POS
<a href="https://raw.githubusercontent.com/JlKmn/ProbingPretrainedLM/main/results/pos_accuracy.png"><img src="https://raw.githubusercontent.com/JlKmn/ProbingPretrainedLM/main/results/pos_accuracy.png" width="600" ></a>

### NER
<a href="https://raw.githubusercontent.com/JlKmn/ProbingPretrainedLM/main/results/ner_recall.png"><img src="https://raw.githubusercontent.com/JlKmn/ProbingPretrainedLM/main/results/ner_recall.png" width="600" ></a>
<a href="https://raw.githubusercontent.com/JlKmn/ProbingPretrainedLM/main/results/ner_precision.png"><img src="https://raw.githubusercontent.com/JlKmn/ProbingPretrainedLM/main/results/ner_precision.png" width="600" ></a>