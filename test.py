import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from Dataset import TextDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = './model_built/model.pth'
tokenizer_path = 'bert-base-uncased'  ####

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=19)
model.load_state_dict(
    torch.load(
        f"./model_built/model.pth",
        map_location="cpu",
    )
)
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)


test_data = pd.read_csv("../test/data/ielts_test.csv") 
test_texts = test_data['Essay'].tolist()
test_labels = test_data['Overall'].tolist()

def map_label_to_class(label):
    return int(label * 2)  # 0, 0.5, 1, 1.5, ..., 9 -> 0, 1, 2, 3, ..., 18

def map_class_to_label(cls):
    return cls / 2.0  # 0, 1, 2, 3, ..., 18 -> 0, 0.5, 1, 1.5, ..., 9

test_encodings = tokenizer(test_texts, truncation=True, padding=True)
print(test_texts)

test_labels = [map_label_to_class(label) for label in test_labels]

test_dataset = TextDataset(test_encodings, test_labels)

model.eval()
predictions = []
with torch.no_grad():
    for batch in test_dataset:
        inputs = {key: val.unsqueeze(0).to(device) for key, val in batch.items() if key != 'labels'}
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())

preds = [map_class_to_label(pred) for pred in predictions]

true_labels = [map_class_to_label(label) for label in test_labels]
accuracy = np.mean(np.array(preds) == np.array(true_labels))

print(f"Accuracy: {accuracy}")

for i in range(len(test_labels)):
    print(f"answer: {map_class_to_label(test_labels[i])}, prediction: {preds[i]}")
