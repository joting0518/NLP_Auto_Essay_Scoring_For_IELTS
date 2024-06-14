import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter
from Dataset import TextDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv("../test/data/ielts_train.csv")
test_data = pd.read_csv("../test/data/ielts_test.csv")
data.head()

def map_label_to_class(label):
    return int(label * 2)  # 0, 0.5, 1, 1.5, ..., 9 -> 0, 1, 2, 3, ..., 18

def map_class_to_label(cls):
    return cls / 2.0  # 0, 1, 2, 3, ..., 18 -> 0, 0.5, 1, 1.5, ..., 9

train_labels = data['Overall'].tolist()
train_texts = data['Essay'].tolist()
test_labels = test_data['Overall'].tolist()
test_texts = test_data['Essay'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_labels = [map_label_to_class(label) for label in train_labels]
test_labels = [map_label_to_class(label) for label in test_labels]

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=19).to(device)

# Define SummaryWriter
writer = SummaryWriter(log_dir='./logs')

class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_step = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        # Log the loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), self.global_step)
        self.global_step += 1

        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='./results/my_model',          
    num_train_epochs=10,             
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=256,   
    warmup_steps=math.ceil((len(data) / 16)) * 10/10,               
    weight_decay=0.01,             
    logging_dir='./logs',          
    logging_steps=10,  
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",  
    logging_strategy="steps" 
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (preds == labels).astype(np.float32).mean().item()
    return {
        'accuracy': acc,
    }

trainer = CustomTrainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=test_dataset,           
    compute_metrics=compute_metrics
)

def save_checkpoint(model, tokenizer, save_path):
    try:
        model.cpu()
        torch.save(model.state_dict(), f"{save_path}/model.pth")
        model.to(device)  # 移回CUDA
        tokenizer.save_pretrained(save_path)
    except Exception as e:
        print(f"存模型有問題 {e}")

trainer.train()
trainer.evaluate()

save_checkpoint(model, tokenizer, './model_built')

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

preds = [map_class_to_label(pred) for pred in preds]

for i in range(len(test_labels)):
    print(f"answer: {map_class_to_label(test_labels[i])}, prediction: {preds[i]}")

# Close the SummaryWriter
writer.close()
