
import os
import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_scheduler
from datasets import load_dataset
from utils import get_device, preprocess_data, ClassificationDataset

def main(data_path, teacher_model_name, student_model_name):
    # Parameters
    max_length = 256
    batch_size = 16
    epochs = 3
    learning_rate = 2e-5

    # Load Dataset
    dataset = load_dataset('csv', data_files={'train': data_path})

    # Tokenizers
    teacher_tokenizer = BertTokenizer.from_pretrained(teacher_model_name)
    student_tokenizer = BertTokenizer.from_pretrained(student_model_name)

    # Tokenize Data
    tokenized_datasets = dataset.map(lambda x: preprocess_data(teacher_tokenizer, x, max_length), batched=True)

    # DataLoader
    train_dataset = ClassificationDataset(tokenized_datasets['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Models
    device = get_device()
    teacher_model = BertForSequenceClassification.from_pretrained(teacher_model_name)
    student_model = BertForSequenceClassification.from_pretrained(student_model_name)
    teacher_model.to(device)
    student_model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(student_model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Distillation Function
    def distill_epoch(teacher_model, student_model, data_loader, optimizer, device, scheduler):
        teacher_model.eval()
        student_model.train()
        total_loss = 0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = student_outputs.loss + torch.nn.functional.kl_div(student_outputs.logits, teacher_outputs.logits)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    # Training Loop
    for epoch in range(epochs):
        train_loss = distill_epoch(teacher_model, student_model, train_loader, optimizer, device, lr_scheduler)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss}')

    # Save Student Model
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    student_model.save_pretrained(model_dir)
    student_tokenizer.save_pretrained(model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing distillation data')
    parser.add_argument('--teacher_model_name', type=str, required=True, help='Name of the teacher model')
    parser.add_argument('--student_model_name', type=str, required=True, help='Name of the student model')
    args = parser.parse_args()
    main(args.data_path, args.teacher_model_name, args.student_model_name)
