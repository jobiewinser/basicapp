import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# Define a dataset class
class MockDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Mock data
texts = ["This is a scientific statement.", "Another science sentence.", "More data needed.", "Testing eval dataset."]
labels = [0, 1, 0, 1]  # Adjust as needed for real binary classification task

# Split into training and evaluation datasets
train_texts = texts[:2]
train_labels = labels[:2]
eval_texts = texts[2:]
eval_labels = labels[2:]

train_dataset = MockDataset(train_texts, train_labels, AutoTokenizer.from_pretrained("bert-base-uncased"))
eval_dataset = MockDataset(eval_texts, eval_labels, AutoTokenizer.from_pretrained("bert-base-uncased"))

# Load pre-trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Training setup
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-5,
    save_total_limit=3,
    eval_strategy="epoch",  # Evaluate at each epoch
    save_strategy="epoch",  # Save model at each epoch
    load_best_model_at_end=True,
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add the evaluation dataset
)

# Train the model
trainer.train()

# Saving the model and tokenizer
model.save_pretrained("./model_output")
tokenizer.save_pretrained("./model_output")

# Inference function
def predict_confidence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to the correct device
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    confidence = scores.max().item()
    return confidence

# Example usage
if __name__ == "__main__":
    statement = "A scientific statement to classify."
    confidence = predict_confidence(statement)
    print(f"Confidence score: {confidence:.2f}")