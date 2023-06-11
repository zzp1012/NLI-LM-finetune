import os
import argparse
import random
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# import internal libs
from data import NLIDataset
from model import BERTNLIClassifier
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.avgmeter import MetricTracker

def evaluate(model: nn.Module,
             dataloader: DataLoader,
             device: torch.device,) -> float:
    """evaluate the model

    Args:
        model (nn.Module): the model to evaluate
        dataloader (DataLoader): the dataloader
        device (torch.device): GPU or CPU
    
    Returns:
        float: the accuracy
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, entries in enumerate(tqdm(dataloader)):
            ids, input_ids, types, masks, labels = entries.values()
            # set the inputs to device
            input_ids, types, masks, labels = \
                input_ids.to(device), types.to(device), masks.to(device), labels.to(device)
            # set the outputs
            outputs = model(input_ids, types, masks)
            # compute the accuracy
            correct += (outputs.max(1)[1] == labels).sum().item()
            total += len(labels)
    return correct / total


def train(save_path: str,
          device: torch.device,
          model: nn.Module,
          trainset: Dataset,
          testset: Dataset,
          optim: str,
          epochs: int,
          lr: float,
          batch_size: int,
          warmup_steps: int,
          steps_saving: int,
          seed: int) -> None:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        trainset: the train dataset
        testset: the test dataset
        optim: the optimizer
        epochs: the epochs
        lr: the learning rate
        batch_size: the batch size
        steps_saving: the steps to save the model
        seed: the seed
    """
    logger = get_logger(__name__)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## set up the basic component for training
    # put the model to GPU or CPU
    model = model.to(device)
    # set the optimizer
    if optim == "AdamW":
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=lr, eps=1e-6, correct_bias=False)
    else:
        raise ValueError(f"optimizer should be SGD or Adam but got {optim}")
    
    # set the scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )
    
    # set the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    ## set up the data part
    # set the trainset loader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # set the testset loader 
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # set the metric tracker
    tracker = MetricTracker()

    for epoch in range(1, epochs+1):
        logger.info(f"######Epoch - {epoch}")
        # train the model
        model.train()
        for batch_idx, entries in enumerate(tqdm(trainloader)):
            _, input_ids, types, masks, labels = entries.values()
            # set the inputs to device
            input_ids, types, masks, labels = \
                input_ids.to(device), types.to(device), masks.to(device), labels.to(device)
            # set the outputs
            outputs = model(input_ids, types, masks)
            # set the loss
            losses = loss_fn(outputs, labels)
            loss = torch.mean(losses)
            # set zero grad
            optimizer.zero_grad()
            # set the loss
            loss.backward()
            # set the optimizer
            optimizer.step()
            # update the scheduler
            scheduler.step()
            # set the loss and accuracy
            tracker.update({
                "train_loss": loss.item(),
                "train_acc": (outputs.max(1)[1] == labels).float().mean().item()
            }, n = input_ids.size(0))

        # save the results
        if epoch % steps_saving == 0 or epoch == epochs:
            torch.save(model.state_dict(), 
                       os.path.join(save_path, f"model_epoch{epoch}.pt"))
            tracker.save_to_csv(os.path.join(save_path, f"train.csv"))

    # evaluate the model
    logger.info("######Evaluate the model on trainset")
    acc = evaluate(model, trainloader, device)
    logger.info(f"Trainset Accuracy: {acc}")
        
    # predict the testset
    model.eval()
    with torch.no_grad():
        id_lst, pred_lst = [], []
        for batch_idx, entries in enumerate(tqdm(testloader)):
            ids, input_ids, types, masks = entries.values()
            # set the inputs to device
            input_ids, types, masks = \
                input_ids.to(device), types.to(device), masks.to(device)
            # set the outputs
            outputs = model(input_ids, types, masks)
            # get the prediction
            preds = outputs.max(1)[1]
            # save the id and prediction
            id_lst.extend(ids)
            pred_lst.extend(preds.tolist())
    
    # save the prediction
    # using pandas 
    df = pd.DataFrame({"Id": id_lst, "Category": pred_lst})
    df["Category"] = df["Category"].map({
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    })
    df.to_csv(os.path.join(save_path, f"predication.csv"), index=False)


def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="/home/zhouzhanpeng/NLI-LM-finetune/outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--data_root", default="/home/zhouzhanpeng/NLI-LM-finetune/data", type=str,
                        help='the path of loading data.')
    parser.add_argument("--resume_path", default=None, type=str,
                        help='the path of loading model.')
    parser.add_argument("--optimizer", default="AdamW", type=str,
                        help='the optimizer name.')
    parser.add_argument('--epochs', default=10, type=int,
                        help="set iteration number")
    parser.add_argument("--lr", default=1e-5, type=float,
                        help="set the learning rate.")
    parser.add_argument("--bs", default=16, type=int,
                        help="set the batch size")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="set the warmup steps")
    parser.add_argument("--steps_saving", default=1, type=int,
                        help="set the steps to save the model")
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.optimizer}",
                         f"epochs{args.epochs}",
                         f"lr{args.lr}",
                         f"bs{args.bs}",])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args


def main():
    # get the args.
    args = add_args()
    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, args.verbose)
    # set the seed
    set_seed(args.seed)
    # set the device
    args.device = set_device(args.device)
    # save the current src
    save_current_src(save_path = args.save_path)

    # show the args.
    logger.info("#########parameters settings....")
    log_settings(args)

    # prepare the dataset
    logger.info("#########preparing dataset....")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    trainset = NLIDataset(
        path=os.path.join(args.data_root, "train.tsv"),
        tokenizer=tokenizer,
    )
    testset = NLIDataset(
        path=os.path.join(args.data_root, "test.tsv"),
        tokenizer=tokenizer,
    )

    # prepare the model
    logger.info("#########preparing model....")
    model = BERTNLIClassifier(num_classes=3)
    if args.resume_path is not None:
        model.load_state_dict(torch.load(args.resume_path))
    logger.info(model)

    # train the model
    logger.info("#########training model....")
    train(save_path = os.path.join(args.save_path, "train"),
          device = args.device,
          model = model,
          trainset = trainset,
          testset = testset,
          optim = args.optimizer,
          epochs = args.epochs,
          lr = args.lr,
          batch_size = args.bs,
          warmup_steps = args.warmup_steps,
          steps_saving = args.steps_saving,
          seed = args.seed)

if __name__ == "__main__":
    main()