
# Language Model Distillation

Welcome to the Language Model Distillation project! This project focuses on distilling a large language model into a smaller model.

## Introduction

Model distillation involves training a smaller model (student) to mimic the behavior of a larger model (teacher). In this project, we leverage the power of BERT models to perform distillation using a dataset of labeled text.

## Dataset

For this project, we will use a custom dataset of labeled text. You can create your own dataset and place it in the `data/distillation_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- Scikit-learn

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/your-username/language_model_distillation.git
cd language_model_distillation

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes labeled text. Place these files in the data/ directory.
# The data should be in a CSV file with two columns: text and label.

# To distill the teacher model into the student model, run the following command:
python scripts/train.py --data_path data/distillation_data.csv --teacher_model_name bert-large-uncased --student_model_name bert-base-uncased

# To evaluate the performance of the distilled model, run:
python scripts/evaluate.py --model_path models/ --data_path data/distillation_data.csv
