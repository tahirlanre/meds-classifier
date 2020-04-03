import logging
import argparse
import random

import numpy as np
import torch
import io
import time
import datetime
import sys

from transformers import BertModel, BertForSequenceClassification

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from tqdm import tqdm

from model.model import BERTGRUModel
from data_loader.dataset import MedsDataset
from data_loader.dataset_sampler import ImbalancedDatasetSampler

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
        #sampler=ImbalancedDatasetSampler(train_data_),
        num_workers = args.num_workers
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.lr_gamma)
    global_step = 0
    #model.zero_grad()
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
                #outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=label)
                output = model(input_ids, input_mask)
                loss = criterion(output, label)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()

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
                
        # Adjust the learning rate
        scheduler.step()

        # Test the model on valid set
        valid_acc , valid_loss, eval_metrics = test(valid_data_)
        writer.add_scalar('Loss/val', valid_loss, global_step)
        writer.add_scalar('Accuracy/val', valid_acc, global_step)
        writer.add_scalar('F1/ Pos class - val', eval_metrics['F1 - Pos class'], global_step)

        print("Test - Accuracy: {}, Loss: {}".format(valid_acc, valid_loss))
        for metric, value in eval_metrics.items():
            if metric == 'Confusion matrix':
                print(f'TP: {value[3]}, FP: {value[1]}, FN: {value[2]}')
            else:
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
            output = model(input_ids, input_mask)
            accuracy = (output.argmax(1) == label).float().mean().item()
            loss = criterion(output, label)
            total_accuracy.append(accuracy)
            total_loss.append(loss.item())

            # Move predictions and labels to CPU and store
            y_true.append(label.to('cpu').numpy())
            y_pred.append(output.argmax(dim=1).detach().cpu().numpy())

    # In case that nothing in the dataset
    if total_accuracy == []:
        return 0.0

    #combine the results across all batches
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    # Calculate evaluation metrics
    f1 = f1_score(y_true, y_pred, average=None)
    f1_avg = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred).ravel()

    eval_metrics = {
              'F1 - Pos class': f1[1],
              'F1 - Neg class': f1[0],
              'F1': f1_avg, 
              'Precision': precision,
              'Recall': recall,
              'Confusion matrix': confusion_mat
            }

    return sum(total_accuracy) / len(total_accuracy), sum(total_loss) / len(total_loss), eval_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Medication mention classification model')
    parser.add_argument('train_data_path', help='path for train data')
    parser.add_argument('valid_data_path', help='path for valid data')
    parser.add_argument('--num-epochs', type=int, default=4,
                        help='num epochs (default=4)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size (default=16)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='learning rate (default=2e-5)')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                         help='Epsilon for Adam optimizer (default=1e-8)')
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
    args = parser.parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    device = args.device
    max_seq_len = args.max_seq_length

    train_data_path = args.train_data_path
    valid_data_path = args.valid_data_path

    logging.basicConfig(level=getattr(logging, args.logging_level))

    logging.info("Loading datasets")
    train_dataset = MedsDataset(train_data_path, max_seq_len)
    valid_dataset = MedsDataset(valid_data_path, max_seq_len)

    # Train!
    logger.info("***** Running training *****")
    logging.info(" Num of train examples = %d", len(train_dataset))
    logging.info(" Num of valid examples = %d", len(valid_dataset))

    train_len = len(train_dataset)

    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    writer = SummaryWriter(f'runs/train/{current_time}')

    logging.info("Creating models")
    #bert = BertModel.from_pretrained("bert-base-uncased")
    # load pre trained model
    bert = BertForSequenceClassification.from_pretrained('saved/models')
    
    # freeze bert parameters so that the gradients are not computed
    for param in bert.parameters():
        param.requires_grad = False 

    HIDDEN_DIM = 256
    OUTPUT_DIM = 2
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model = BERTGRUModel(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT).to(device)

    logging.info("Starting training")
    train_and_valid(lr, num_epochs, train_dataset, valid_dataset)

    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.to('cpu'), args.save_model_path)



