import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = './model_built/model.pth'
tokenizer_path = 'bert-base-uncased'

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=19)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

def map_label_to_class(label):
    return int(label * 2)  # 0, 0.5, 1, 1.5, ..., 9 -> 0, 1, 2, 3, ..., 18

def map_class_to_label(cls):
    return cls / 2.0  # 0, 1, 2, 3, ..., 18 -> 0, 0.5, 1, 1.5, ..., 9

import google.generativeai as genai
import os

api_key = ''
genai.configure(api_key=api_key)

def predict_score(text):
    model.eval()
    with torch.no_grad():
        word_count = len(text.split())
        encodings = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in encodings.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        score = map_class_to_label(pred)
        
        model_api = genai.GenerativeModel('gemini-pro')
        response = model_api.generate_content('give advice to this essay to improve its score. Essay:' + text)
        feedback = response.text
        
        return score, word_count, feedback

iface = gr.Interface(
    fn=predict_score,
    inputs=gr.Textbox(lines=5, placeholder="Enter your essay here..."),
    outputs=[
        gr.Number(label="Your Score"),
        gr.Number(label="Word Count"),
        gr.Textbox(label="Feedback")
    ],
    title="Auto IELTS Essay Scoring System",
    description="Enter your essay to get a score and feedback."
)

if __name__ == "__main__":
    iface.launch() # iface.launch(share=True)
