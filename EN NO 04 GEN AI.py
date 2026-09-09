import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import evaluate

print("All required libraries are installed.")

dataset = load_dataset("imdb")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

small_train = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
small_test = tokenized_dataset["test"].shuffle(seed=42).select(range(500))

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_test,
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate()
print(results)

classifier = pipeline(
    "sentiment-analysis",
    model=trainer.model,
    tokenizer=tokenizer
)

print(classifier("The movie was absolutely wonderful!"))
print(classifier("The film was boring and disappointing."))

sentences = [
    "The movie was absolutely wonderful!",
    "The film was boring and disappointing.",
    "I love this product.",
    "The service was terrible.",
    "The food was delicious.",
    "The laptop performance is excellent.",
    "I am not satisfied with the quality.",
    "This is the best experience I have ever had."
]

predictions = classifier(sentences)

for sentence, prediction in zip(sentences, predictions):
    print("Sentence :", sentence)
    print("Prediction:", prediction)
    print("-" * 60)