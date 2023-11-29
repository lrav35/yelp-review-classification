import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from tqdm.auto import tqdm
import evaluate
import uuid



class ReviewDataSet(Dataset):
    """ torch Dataset for our review data """
    def __init__(self, tokenizer, dataframe):
        self.tokenizer = tokenizer
        self.data = self.format_dataset(dataframe)
        self.size = len(dataframe)
        

    """gives length of dataset"""
    def __len__(self) -> int:
        return self.size

    """function allows object to be subscriptable -> dataset[0]"""
    def __getitem__(self, idx: int):
        label = self.data['labels'][idx]
        input_ids = self.data['input_ids'][idx]
        token_type = self.data['token_type_ids'][idx]
        att_mask = self.data['attention_mask'][idx]
        sample = {'labels': label, 'input_ids': input_ids, 'token_type_ids': token_type, 'attention_mask': att_mask}
        return sample


    def tokenize_text(self, text_data):
        token_dict = self.tokenizer(text_data, padding='max_length', truncation=True)
        input = torch.tensor(token_dict['input_ids'])
        token_type = torch.tensor(token_dict['token_type_ids'])
        att_mask = torch.tensor(token_dict['attention_mask'])
        return input, token_type, att_mask

    def format_dataset(self, dataframe: pd.DataFrame):
        input, token_type, att_mask = zip(*dataframe['text'].apply(self.tokenize_text))
        self.data = {
            'labels': torch.tensor(dataframe['stars'].to_list())-1,
            'input_ids': input,
            'token_type_ids': token_type,
            'attention_mask': att_mask
        }
        return self.data


def load_and_prepare_data(path: str, sample_size: int, COLAB: bool) -> pd.DataFrame:
    if COLAB:
        from google.colab import drive
        drive.mount('/content/drive')

    # read data
    data = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            data.append(json.loads(line))
    df = pd.DataFrame(data)

    # create dataframe
    review_df = df[['stars', 'text']].astype({'stars': int})

    # randomly sample from each class based on how many samples the minority class holds
    seed = 42
    minority_class_count = len(review_df[review_df.loc[:, ('stars')] == 2])
    review_df = review_df.groupby('stars').apply(lambda x: x.sample(n=minority_class_count, random_state=seed)).reset_index(drop=True)
    return review_df

def should_run_eval(total_steps, freq, current_step):
    return current_step % (total_steps // freq) == 0

def eval(model, val_data):
    print("evaluating model...\n")
    metric = evaluate.load("accuracy")
    preds_and_true = {'preds': [], 'labels': []}
    model.eval()
    for batch in val_data:
        batch = {
            "input_ids": batch["input_ids"].to(model.device),
            "labels": batch["labels"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
        }
        with torch.no_grad():
            outputs = model(**batch)
        
        # record loss
        val_loss = outputs.loss.item()

        # compute accuracy
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        metric.add_batch(predictions=preds, references=batch["labels"])
        preds_and_true['preds'].extend([p.item() for p in preds+1])
        preds_and_true['labels'].extend([l.item() for l in batch["labels"]+1])

        # pbar.update(1)
    acc_result = metric.compute()
    print(f"Accuracy: {acc_result['accuracy']}")
    return acc_result['accuracy'], preds_and_true, val_loss

def save_model(model, outpath: str, current_epoch: int, current_step: int, results: dict):
    print(f"saving model at epoch: {current_epoch}, step: {current_step}")
    outpath += f"/epoch_{current_epoch}/step_{current_step}"
    model.save_pretrained(outpath)
    confusion(results['labels'], results['preds'], current_epoch, current_step, outpath)
    

def confusion(true: list, pred: list, epoch, step, outpath):
    ConfusionMatrixDisplay.from_predictions(true, pred)
    outpath += f'/confusion_{epoch}_{step}.jpeg'
    plt.savefig(outpath)
    plt.close()

def generate_loss_image(train_loss: list, val_loss: list, output_dir: str):
    iter_x_ind = np.linspace(0, len(val_loss)-1, num=len(train_loss))
    interp_val_loss = np.interp(iter_x_ind, np.arange(len(val_loss)), val_loss)
    plt.plot(np.array(train_loss), color='b', label='training loss')
    plt.plot(interp_val_loss, color='r', label='validation loss')
    plt.savefig(output_dir + '/loss_plot.jpeg')

def train_model(model, epochs, train_dataloader, val_dataloader, train_steps, optimizer, lr_scheduler, save_path: str):
    pbar = tqdm(range(train_steps))

    run_id = str(uuid.uuid4())
    print(f"model id :: {run_id}")
    output_dir = f"{save_path}/outputs/bert/{run_id}"
    model.train()
    best_accuracy = 0.0
    train_epoch_loss = []
    val_epoch_loss = []
    for epoch in range(epochs):
        current_epoch = epoch + 1
        train_batch_loss = []
        for step, batch in enumerate(train_dataloader):
            current_step = step + 1
            pbar.set_description(f"Epoch {current_epoch} :: Step {current_step}")

            batch = {
                "input_ids": batch["input_ids"].to(model.device),
                "labels": batch["labels"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
            }

            # forward
            outputs = model(**batch)
            loss = outputs.loss

            train_batch_loss.append(loss.item())

            # backward
            loss.backward()

            # update weights
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # evaluate and save model
            if should_run_eval(len(train_dataloader), 2, current_step):
                accuracy, results, val_loss = eval(model, val_dataloader)
                val_epoch_loss.append(val_loss)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    save_model(model, output_dir, current_epoch, current_step, results)
                else:
                    print('skipping model save...')
                print(f"current best accuracy: {best_accuracy}\n")
                model.train()
            pbar.update(1)
        train_epoch_loss.extend(train_batch_loss)
    generate_loss_image(train_epoch_loss, val_epoch_loss, output_dir)

if __name__ == '__main__':

    # load data
    print("loading data...\n")
    path = "yelp_academic_dataset_review.json"
    df = load_and_prepare_data(path, COLAB=False)

    # create dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=False)
    review_dataset = ReviewDataSet(tokenizer, df)

    # train and val split
    train_size = int(0.8 * len(review_dataset))
    val_size = len(review_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(review_dataset, [train_size, val_size], generator)
    print(f'train and validation dataset sizes: {len(train_dataset), len(val_dataset)}\n')

    # dataloader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    # initialize model
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=5)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_params = sum([p.numel() for p in model.parameters()])
    print(f"model params: {num_params}\n")

    epochs = 3
    train_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=train_steps)

    # set device and send model to device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    model.to(device)

    print("beginning model training...\n")
    train_model(model, epochs, train_dataloader, val_dataloader)

    print("complete")