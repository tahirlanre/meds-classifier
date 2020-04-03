import logging
import argparse
import random

import numpy as np
import torch
import io
import time
import datetime
import sys

from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from tqdm import tqdm

from data_loader.dataset import MedsDataset

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def train_and_valid(lr_, num_epoch, train_data_, valid_data_):
    r"""
    Here we use Adam optimizer to train the model.

    Arguments:
        lr_: learning rate
        num_epoch: the number of epoches for training the model
        train_data_: the data used to train the model
        valid_data_: the data used to validation
    """
    train_data = DataLoader(
        train_data_,
        batch_size=batch_size,
        num_workers = args.num_workers
        )
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    total_steps = len(train_data) * num_epochs

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    global_step = 0
    # Train!
    model.zero_grad()
    for epoch in range(num_epoch):
        print("Training on epoch {}".format(epoch))
        # Train the model
        with tqdm(unit_scale=0, unit='lines', total=train_len) as t:
            avg_loss = 0.0
            for i, (text, attention_mask, label) in enumerate(train_data):
                global_step += 1
                model.train()
                t.update(len(label))
                optimizer.zero_grad()
                input_ids, input_mask, label = text.to(device), attention_mask.to(device), label.to(device)
                outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=label)
                loss = outputs[0]
                loss.backward()
                avg_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()

                # Log metrics
                if i % (16 * batch_size) == 0:
                    avg_loss = avg_loss / (16 * batch_size)
                    #writer.add_scalar('Loss/train', loss.item(), global_step)
                    writer.add_scalar('Loss/train', avg_loss, global_step)
                    writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    avg_loss = 0
                    t.set_description(
                        "lr: {:9.5f} loss: {:9.5f}".format(
                            scheduler.get_last_lr()[0], loss))

        # Test the model on valid set
        valid_acc , valid_loss, eval_metrics = test(valid_data_)

        # remember best acc@1 and save checkpoint
        is_best = valid_acc > best_acc1
        if is_best:
            model.save_pretrained('saved/models')

        writer.add_scalar('Loss/val', valid_loss, global_step)
        writer.add_scalar('Accuracy/val', valid_acc, global_step)
        
        print("Test - Accuracy: {}, Loss: {}".format(valid_acc, valid_loss))
        for metric, value in eval_metrics.items():
            print(f'{metric}: {value}')

def test(data_):
    r"""
    Arguments:
        data_: the data used to train the model
    """
    data = DataLoader(
        data_,
        batch_size=batch_size,
        num_workers=args.num_workers)
    total_accuracy = []
    total_loss = []
    y_true = []
    y_pred = []
    for text, attention_mask, label in data:
        model.eval()
        with torch.no_grad():
            input_ids, input_mask, label = text.to(device), attention_mask.to(device), label.to(device)
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=label)
            loss = outputs[0]
            accuracy = (outputs[1].argmax(1) == label).float().mean().item()
            total_accuracy.append(accuracy)
            total_loss.append(loss.item())

            # Move predictions and labels to CPU and store
            y_true.append(label.to('cpu').numpy())
            y_pred.append(outputs[1].argmax(dim=1).detach().cpu().numpy())

    # In case that nothing in the dataset
    if total_accuracy == []:
        return 0.0

    #combine the results across all batches
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    # Calculate evaluation metrics to compare with benchmark from authors
    F1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    eval_metrics = {
        'F1': F1, 
        'Precision': precision,
        'Recall': recall
    }

    return sum(total_accuracy) / len(total_accuracy), sum(total_loss) / len(total_loss), eval_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Medication mention classification model')
    parser.add_argument('data_path', help='path for train data')
    parser.add_argument('--num-epochs', type=int, default=4,
                        help='num epochs (default=4)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size (default=16)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='learning rate (default=2e-5)')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                         help='Epsilon for Adam optimizer (default=1e-8)')
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--lr-gamma', type=float, default=0.9,
                        help='gamma value for lr (default=0.9)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='num of workers (default=1)')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='The maximum total input sequence length after tokenization. Sequences longer' 
                        'than this will be truncated, sequences shorter will be padded (default=128)')
    parser.add_argument('--device', default='cuda',
                        help='device (default=cuda)')
    parser.add_argument('--save-model-path',
                        help='path for saving model')
    parser.add_argument('--logging-level', default='WARNING',
                        help='logging level (default=WARNING)')
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--split-ratio', type=float, default=0.7,
                        help='train/valid split ratio (default=0.7)')

    args = parser.parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    device = args.device
    max_seq_len = args.max_seq_length
    split_ratio = args.split_ratio
    data_path = args.data_path

    logging.basicConfig(level=getattr(logging, args.logging_level))

    logging.info("Loading datasets")
    train_dataset = MedsDataset(data_path, max_seq_len)

    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/pre-train/{current_time}')
    # split train_dataset into train and valid
    train_len = int(len(train_dataset) * split_ratio)
    sub_train_, sub_valid_ = \
        random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    logging.info("Creating models")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                            output_hidden_states=True,
                                                            output_attentions=True).to(device)

    best_acc1 = 0
    logging.info("Starting training")
    train_and_valid(lr, num_epochs, sub_train_, sub_valid_)
    

