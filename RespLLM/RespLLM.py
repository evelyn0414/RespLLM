import numpy as np
from transformers import AutoTokenizer
import transformers
import torch
from torch import nn
from torch.nn import functional as F
import os
import time
import collections
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

from src.benchmark.llm_eval.model import RespLLM
from src.benchmark.llm_eval.util import test, get_dataloader, EarlyStopper, set_all_seed
from src.benchmark.llm_eval.sampler import CategoriesSampler


# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.loggers import CSVLogger
# from lightning.pytorch import seed_everything
# from torchmetrics import AUROC

token = "redacted"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_RespLLM(configs):
    train_loaders, val_loaders, test_loaders = [], [], []
    n_cls = collections.defaultdict(lambda:2, {"11": 5, "12": 5})
    for task in configs.train_tasks:
        train_loader, val_loader, _ = get_dataloader(configs, task)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    
    for task in configs.test_tasks:
        _, _, test_loader = get_dataloader(configs, task)
        test_loaders.append(test_loader)
    

    time_now = time.time()
    train_steps = len(train_loader)
    model = RespLLM(configs).to(DEVICE)
    print(model.llm_model)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = torch.optim.Adam(trained_parameters, lr=configs.lr)
    loss_func = nn.CrossEntropyLoss()

    early_stopper = EarlyStopper(patience=2, min_delta=0.01)
    

    for epoch in tqdm(range(configs.train_epochs)):
        iter_count = 0
        train_loss = []

        iterators = [iter(dataloader) for dataloader in train_loaders]
        num_batch = [len(dataloader) for dataloader in train_loaders]

        model.train()
        epoch_time = time.time()

        i = 0
        while True:
            i += 1
            # Randomly select a dataloader
            # selected_idx = random.randint(0, len(configs.train_tasks) - 1)
            selected_idx = random.choices(range(len(configs.train_tasks)), weights=num_batch, k=1)[0]
            selected_iterator = iterators[selected_idx]
            
            try:
                # Get the next batch from the selected dataloader
                x1, x2, x3, y = next(selected_iterator)
            except StopIteration:
                # If any iterator is exhausted, break the loop
                break
            
            
            x1 = x1.to(DEVICE)
            y = y.to(DEVICE)
            iter_count += 1
            model_optim.zero_grad()

            y_hat = model(x1, x2, x3)
            loss = loss_func(y_hat, y)
            train_loss.append(loss.item())

            if i < 3 or (i + 1) % 10 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((configs.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            
            loss.backward()
            model_optim.step()
        
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        print("train loss", train_loss)


        model.eval()

        # train
        print("="*10 + "train set eval")
        for j, train_loader in enumerate(train_loaders):
            test(model, train_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.train_tasks[j] + "train")
        
        # validation
        print("="*10 + "validation")
        validation_loss = 0
        for j, val_loader in enumerate(val_loaders):
            validation_loss += test(model, val_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.train_tasks[j] + "val")
        
        if (epoch + 1) % configs.test_interval == 0:
            print("="*10 + "test")
            for j, test_loader in enumerate(test_loaders):
                # test
                print("Task", configs.test_tasks[j])
                if  n_cls[configs.test_tasks[j]] != configs.n_cls:
                    # test(model, test_loader, loss_func, configs.n_cls, plot_feature="llm/task" + configs.test_tasks[j] + "test", plot_only=True)
                    pass
                else:
                    test(model, test_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.test_tasks[j] + "test")

        if (epoch + 1) % configs.meta_val_interval == 0:
            configs.save_pth = f"cks/llm/model_{configs.llm_model}_{epoch+1}epoch.pt"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 'loss': validation_loss,
                }, configs.save_pth)
            
        # do_extract_and_evaluate(model, configs, train_loader=train_loader, que)
        if early_stopper.early_stop(validation_loss):
            print("early stopping")      
            break
    
            
    configs.save_pth = f"cks/llm/model_{configs.llm_model}_{epoch+1}epoch.pt"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'loss': validation_loss,
        }, configs.save_pth)
    


def evaluate_RespLLM(configs):
    train_loaders, val_loaders, test_loaders = [], [], []
    n_cls = collections.defaultdict(lambda:2, {"11": 5, "12": 5})

    for task in configs.train_tasks:
        train_loader, val_loader, _ = get_dataloader(configs, task)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    
    for task in configs.test_tasks:
        _, _, test_loader = get_dataloader(configs, task)
        test_loaders.append(test_loader)


    checkpoint = torch.load(configs.save_pth)
    model = RespLLM(configs).to(DEVICE)
    # print(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    loss_func = nn.CrossEntropyLoss()

    # # train
    # print("="*10 + "train set eval")
    # for j, train_loader in enumerate(train_loaders):
    #     for i in range(configs.n_run_finetune):
    #         auc = test(model, train_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.train_tasks[j] + "train")
    
    # # validation
    # print("="*10 + "validation")
    # validation_loss = 0
    # for j, val_loader in enumerate(val_loaders):
    #     validation_loss += test(model, val_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.train_tasks[j] + "val")

    print("="*10 + "test")
    for j, test_loader in enumerate(test_loaders):
        # test
        print("Task", configs.test_tasks[j])
        if n_cls[configs.test_tasks[j]] != configs.n_cls:
            pass
        else:
            test(model, test_loader, loss_func, configs.n_cls)




if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    parser.add_argument("--n_cls", type=int, default=2)

    parser.add_argument("--train_tasks", type=str, default="1,2,4.1,4.2,4.5,4.6,7")
    parser.add_argument("--test_tasks", type=str, default="1,2,4.1,4.2,4.5,4.6,7,8,8.1,8.2,8.3,8.5,8.6,8.7,8.8,10,10.5,10.6")

    parser.add_argument("--audio_encoder", type=str, default="operaCT")
    parser.add_argument("--enc_dim", type=int, default=768)
    parser.add_argument("--audio_peft", type=str, default='frozen') # frozen, lora, IA3, full
    parser.add_argument("--audio_lora_rank",  type=int, default=8)

    parser.add_argument("--llm_model", type=str, default='llama')
    # parser.add_argument('--patch_nums', type=int, default=1, help='dimension of fcn') 
    # parser.add_argument('--d_ff', type=int, default=4096, help='dimension of fcn') 
    parser.add_argument('--patch_nums', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument("--llm_dim",  type=int, default=4096)
    parser.add_argument("--llm_peft", type=str, default='lora') # frozen, lora, IA3, full
    parser.add_argument("--llm_lora_rank",  type=int, default=16)
    parser.add_argument("--llm_lora_alpha",  type=int, default=32)
    parser.add_argument("--llm_lora_dropout",  type=float, default=0.1)
    parser.add_argument("--llm_lora_allproj",  type=bool, default=False)

    parser.add_argument("--aligner",  type=str, default="projection") # projection, Qformer, CNN, reprogram

    parser.add_argument("--head", type=str, default="linear")
    parser.add_argument("--head_dropout",  type=float, default=0)

    parser.add_argument("--n_run", type=int, default=5)
    parser.add_argument("--n_run_finetune", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_pct", type=float, default=1)
    parser.add_argument("--train_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=True) #type=int, default=1)
    # parser.add_argument("--test_sampler", default=False)
    parser.add_argument("--use_audio", action=argparse.BooleanOptionalAction, default=True)
    
    parser.add_argument("--test_interval", type=int, default=1) #4
    parser.add_argument("--meta_val_interval", type=int, default=3) #4
    parser.add_argument("--meta_val_iter", type=int, default=10)
    parser.add_argument("--meta_val_way", type=int, default=5)
    parser.add_argument("--meta_val_shot", type=int, default=20)
    parser.add_argument("--meta_val_query", type=int, default=15)
    parser.add_argument("--meta_val_metric", type=str, default="euclidean") # euclidean, cosine, l1, l2

    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--use_L2N", type=bool, default=False)
    parser.add_argument("--use_centering", type=bool, default=False)

    parser.add_argument("--from_pretrain", type=bool, default=False)

    parser.add_argument("--save_pth", type=str, default="cks/llm/model_llama.pt")
    parser.add_argument('--eval_fc', action='store_true', help='do evaluate with final fc layer.')

    parser.add_argument("--few_shot_finetuning", type=bool, default=False)
    parser.add_argument("--data_efficient_finetuning", type=bool, default=True)

    args = parser.parse_args()
    print(args)

    args.train_tasks = args.train_tasks.split(",")
    args.test_tasks = args.test_tasks.split(",")

    if args.from_pretrain:
        evaluate_RespLLM(args)
    else:
        train_RespLLM(args)