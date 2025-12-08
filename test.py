import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from torch.optim import AdamW
from evaluate import load
from seq2seq import create_transformers_train_data, train_transformer, decode_with_transformer

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
# Load Metrics
bleu = load("bleu")
bertscore = load("bertscore")

# %%
# Load Data
data = pd.read_csv('yelp_parallel/yelp_parallel/test_en_parallel.txt', sep='\t')
negative = data["Style 1"].values.tolist()
positive = data["Style 2"].values.tolist()


# %%
def run_experiment(model_name, input_data, target_data, lr, epochs, num_examples=10):
    print(f"--- Experiment: {model_name} | LR: {lr} | Epochs: {epochs} ---")

    # 1. Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # 2. Create Data Loader (Using only the first num_examples for this specific test run if desired,
    # but usually we train on a larger set. Here I will train on the full passed data for better results)
    train_dataset = create_transformers_train_data(input_data, target_data, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=data_collator)

    # 3. Initialize Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # 4. Train
    train_transformer(model, train_loader, optimizer, epochs, device=device)

    # 5. Evaluate on 10 sentences
    # print(f"\n--- Evaluation on first {num_examples} examples ---")
    predictions = []
    references = []

    for i in range(num_examples):
        pred = decode_with_transformer(input_data[i], tokenizer, model, device=device)
        ref = target_data[i]

        predictions.append(pred)
        references.append(ref)

        # print(f"Example {i + 1}:")
        # print(f"  Input : {input_data[i]}")
        # print(f"  Pred  : {pred}")
        # print(f"  Ref   : {ref}")
        # print("-" * 20)

    # 6. Compute Metrics for the batch
    bleu_res = bleu.compute(predictions=predictions, references=references)

    # Using lang="en" for speed as requested
    bert_res = bertscore.compute(predictions=predictions, references=references, lang="en")

    # Calculate average BERT F1 score
    avg_bert_f1 = sum(bert_res['f1']) / len(bert_res['f1'])

    print(f"Batch BLEU: {bleu_res['bleu']}")
    print(f"Batch Avg BERT F1: {avg_bert_f1}")
    print("=" * 40)

    return bleu_res, bert_res


# %% md
### 1. T5 - Default Hyperparameters
# %%
t5_res_1 = run_experiment(
    model_name="t5-small",
    input_data=negative,
    target_data=positive,
    lr=0.001,
    epochs=5
)

# %% md
### 2. T5 - Different Hyperparameters (Lower LR)
# %%
t5_res_2 = run_experiment(
    model_name="t5-small",
    input_data=negative,
    target_data=positive,
    lr=0.0001,
    epochs=5
)

# %% md
### 3. FLAN-T5
# %%
flan_res = run_experiment(
    model_name="google/flan-t5-small",
    input_data=negative,
    target_data=positive,
    lr=0.001,
    epochs=5
)