from pathlib import Path
import torch
from tqdm import tqdm
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup


DEVICE = 'cuda'

def accuracy(preds, y):
    preds = torch.argmax(preds, axis=1)
    y = (y == 1).nonzero().squeeze()[:, 1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train_step(model, optimizer, criterion, dataloader):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    model = model.to(DEVICE)
    for batch, labels in tqdm(dataloader):
        batch = batch.int().to(DEVICE)
        labels = labels.float().to(DEVICE)
        preds = model(batch)
        
        acc = accuracy(preds, labels)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def eval_step(model, criterion, dataloader):
    eval_loss = 0
    eval_acc = 0

    with torch.no_grad():
        model.eval()
        model = model.to(DEVICE)
        for batch, labels in dataloader:
            batch = batch.int().to(DEVICE)
            labels = labels.float().to(DEVICE)
            preds = model(batch)

            acc = accuracy(preds, labels)
            loss = criterion(preds, labels)

            eval_loss += loss.item()
            eval_acc += acc.item()

    return eval_loss / len(dataloader), eval_acc / len(dataloader)


def train(model, train_dataloader, test_dataloader, epochs, logdir):
    history = {
        'train': {
            'loss': [],
            'acc': []
        },
        'eval': {
            'loss': [],
            'acc' : []
        }
    }
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    writer = SummaryWriter(logdir)


    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, optimizer, criterion, train_dataloader)
        scheduler.step()
        eval_loss, eval_acc = eval_step(model, criterion, test_dataloader)
        history['train']['loss'].append(train_loss)
        history['train']['acc'].append(train_acc)
        history['eval']['loss'].append(eval_loss)
        history['eval']['acc'].append(eval_acc)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/eval', eval_loss, epoch)
        writer.add_scalar('Accuracy/eval', eval_acc, epoch)
        torch.save(model.state_dict(), Path(logdir, f'model_{epoch}.pt'))
    return history

