
import json
from glob import glob
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.util import train_test_split_from_list, plot_tsne
from src.model.models_eval import LinearHead, LinearHeadR, AudioClassifier
from src.benchmark.linear_eval import FeatureDataset, DecayLearningRate
import collections
from tqdm import tqdm
from src.benchmark.llm_eval.sampler import SplitCategoriesSampler, TrainCategoriesSampler
from lightning.pytorch.utilities import CombinedLoader
import time
from torch import nn
from torch.nn import functional as F
from src.benchmark.llm_eval.util import itr_merge, EarlyStopper, downsample_balanced_dataset, upsample_balanced_dataset, set_all_seed
from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model
import random
from torchmetrics import AUROC
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, from_npy=False,  from_audio=False):
        self.data = data[0]
        self.label = data[1]
        self.from_npy = from_npy
        self.from_audio = from_audio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.from_npy:
            npy_path = self.data[idx]
            x = np.load(npy_path + ".npy")
        else:
            x = self.data[idx]

        label = self.label[idx]

        if self.from_audio:
            return x, label

        x = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return x, label


def get_dataloader(configs, task, deft_seed=None, sample=False):
    tasks_config = {
        "1": ("coviduk", "covid", "exhalation"),
        "2": ("coviduk", "covid", "cough"),
        "4.1": ("covid19sounds", "covid", "breath"),
        "4.2": ("covid19sounds", "covid", "cough"),
        "4.5": ("covid19sounds", "smoker", "breath"),
        "4.6": ("covid19sounds", "smoker", "cough"),
        "7": ("icbhidisease", "copd", "lung"),
        "8.5": ("coswara", "covid", "cough-shallow"),
        "8.6": ("coswara", "covid", "cough-heavy"),
        "8.7": ("coswara", "covid", "breathing-shallow"),
        "8.8": ("coswara", "covid", "breathing-deep"),

        "10.5": ("kauh", "copd", "lung"),
        "10.6": ("kauh", "asthma", "lung"),

    }
    n_cls = collections.defaultdict(lambda:2, {"11": 5, "12": 5})

    dataset, label, modality = tasks_config[task]
    
    feature_dirs = {"covid19sounds": "feature/covid19sounds_eval/downsampled/", 
                    "coswara": "feature/coswara_eval/",
                    "coviduk": "feature/coviduk_eval/",
                    "kauh": "feature/kauh_eval/",
                    "copd": "feature/copd_eval/",
                    "icbhidisease": "feature/icbhidisease_eval/",
                    }
    
    pad_len_htsat = {"covid19sounds": 8.18, 
                    "coswara": 8.18,
                    "coviduk": 8.18,
                    "copd": 8.18,
                    "kauh": 8.18,
                    "icbhidisease": 8.18,
                    }
    feature_dir = feature_dirs[dataset]
    if task in ["4.1", "4.2"]:
        feature_dir = "feature/covid19sounds_eval/covid_eval/"
    elif task in ["4.5", "4.6"]:
        feature_dir = "feature/covid19sounds_eval/smoker_eval/"
    
    if dataset in ["ssbpr", "copd", "kauh", "icbhidisease"]:
        suffix_dataset =  ".npy"
    elif dataset in ["covid19sounds", "coviduk"]:
        suffix_dataset = "_{}.npy".format(modality)
    elif dataset in ["coswara"]:
        suffix_dataset = "_{}_{}.npy".format(modality, label)
    elif dataset in ["coughvid"]:
        suffix_dataset = "_{}.npy".format(label)
    else:
        raise NotImplementedError

    if dataset == "coviduk":
        y_label = np.load(feature_dir + f"label_{modality}.npy")
    elif dataset == "coswara":
        broad_modality = modality.split("-")[0]
        y_label = np.load(feature_dir + "{}_aligned_{}_label_{}.npy".format(broad_modality, label, modality))
    elif dataset == "kauh":
        y_label = np.load(feature_dir + "labels_both.npy")
        if label == "copd":
            label_dict = {"healthy": 0, "asthma": 2, "COPD": 1, "obstructive": 2}
            y_label = np.array([label_dict[y] for y in y_label])
        elif label == "asthma":
            label_dict = {"healthy": 0, "asthma": 1, "COPD": 2, "obstructive": 2}
            y_label = np.array([label_dict[y] for y in y_label])
        else:
            label_dict = {"healthy": 0, "asthma": 1, "COPD": 1, "obstructive": 1}
            y_label = np.array([label_dict[y] for y in y_label])
    elif dataset == "coughvid":
        y_label = np.load(feature_dir + "label_{}.npy".format(label))
    else:
        y_label = np.load(feature_dir + "labels.npy")
    
    from_audio = False
    seed = 42

    if dataset == "icbhidisease":
        x_data = np.load(feature_dir + f"segmented_spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset)
        y_label = np.load(feature_dir + f"segmented_labels.npy")
        # index_sampled = np.load(feature_dir + f"segmented_index.npy")
    else:
        x_data = np.load(feature_dir + f"spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset)
    
    print(len(x_data), len(y_label))
    print(collections.Counter(y_label))

    x_metadata = np.array(["" for x in range(len(y_label))])

    if dataset == "ssbpr":
        train_ratio = 0.6
        validation_ratio = 0.2
        test_ratio = 0.2
        
        _x_train, x_data_test, _x_metadata_train, x_metadata_test, _y_train, y_label_test = train_test_split(
                x_data, x_metadata, y_label, test_size=test_ratio, random_state=seed, stratify=y_label
            )

        x_data_train, x_data_vad, x_metadata_train, x_metadata_vad, y_label_train, y_label_vad = train_test_split(
                _x_train, _x_metadata_train, _y_train, test_size=validation_ratio/(validation_ratio + train_ratio), 
                random_state=seed, stratify=_y_train
            )
    elif dataset == "covid19sounds":
        y_set = np.load(feature_dir + "data_split.npy")
        
        if task in ["4.1", "4.2"]:
            x_data_train = x_data[y_set == "train"]
            x_metadata_train = x_metadata[y_set == "train"]
            y_label_train = y_label[y_set == "train"]
            
            x_data_vad = x_data[y_set == "validation"]
            x_metadata_vad = x_metadata[y_set == "validation"]
            y_label_vad = y_label[y_set == "validation"]

            x_data_test = x_data[y_set == "test"]
            x_metadata_test = x_metadata[y_set == "test"]
            y_label_test = y_label[y_set == "test"]
        else:
            x_data_train = x_data[y_set == 0]
            x_metadata_train = x_metadata[y_set == 0]
            y_label_train = y_label[y_set == 0]
            
            x_data_vad = x_data[y_set == 1]
            x_metadata_vad = x_metadata[y_set == 1]
            y_label_vad = y_label[y_set == 1]

            x_data_test = x_data[y_set == 2]
            x_metadata_test = x_metadata[y_set == 2]
            y_label_test = y_label[y_set == 2]
        
    elif dataset == "coswara":
        if label == "covid":
            set_all_seed(seed)
            # symptoms = np.array([1 if 'following respiratory symptoms' in m else 0 for m in x_metadata])
            symptoms = np.load(feature_dir + f"symptom" + suffix_dataset)

            group1_indices = np.where((y_label == 0) & (symptoms == 1))[0]
            group2_indices = np.where((y_label == 0) & (symptoms == 0))[0]
            group3_indices = np.where((y_label == 1) & (symptoms == 1))[0]
            group4_indices = np.where((y_label == 1) & (symptoms == 0))[0]
            random.seed(seed)

            test_size = np.min([len(group) for group in [group1_indices, group2_indices, group3_indices, group4_indices]]) - (configs.meta_val_shot // 2)

            def sample_indices(group_indices, test_size):
                print(f"sampling {test_size} from", len(group_indices))
                test_sample_indices = np.random.choice(group_indices, size=test_size, replace=False)
                remaining_indices = np.setdiff1d(group_indices, test_sample_indices)
                return test_sample_indices, remaining_indices
        
            # Step 2: Sample 30 indices for each group for the test set
            group1_indices_test, group1_indices_train = sample_indices(group1_indices, test_size)
            group2_indices_test, group2_indices_train = sample_indices(group2_indices, test_size)
            group3_indices_test, group3_indices_train = sample_indices(group3_indices, test_size)
            group4_indices_test, group4_indices_train = sample_indices(group4_indices, test_size)

            # Combine test and training indices
            indices_test = np.concatenate([group1_indices_test, group2_indices_test, group3_indices_test, group4_indices_test])
            indices_train = np.concatenate([group1_indices_train, group2_indices_train, group3_indices_train, group4_indices_train])

            print("train")
            for indices_array in [group1_indices_train, group2_indices_train, group3_indices_train, group4_indices_train]:
                print(len(indices_array), end=";")
            print("\ntest")
            for indices_array in[group1_indices_test, group2_indices_test, group3_indices_test, group4_indices_test]:
                print(len(indices_array), end=";")
            print()
            # Step 3: Use the sampled indices to get the test and training data
            x_data_test = x_data[indices_test]
            x_metadata_test = x_metadata[indices_test]
            y_label_test = y_label[indices_test]
            symptoms_test = symptoms[indices_test]

            x_data_train = x_data[indices_train]
            x_metadata_train = x_metadata[indices_train]
            y_label_train = y_label[indices_train]
            symptoms_train = symptoms[indices_train]

            x_data_vad, x_metadata_vad, y_label_vad = x_data_train, x_metadata_train, y_label_train 

            group_idxs = []
            for i in range(len(x_data_train)):
                y = y_label_train[i]
                m = symptoms_train[i]
                if y == 0 and m == 1:
                    group = 1
                if y == 0 and m == 0:
                    group = 2
                if y == 1 and m == 1:
                    group = 3
                if y == 1 and m == 0:
                    group = 4
                group_idxs.append(group)
        
            group_idxs = np.array(group_idxs)
        
        else:
            # smoker
            _x_train, x_data_test, _x_metadata_train, x_metadata_test, _y_train, y_label_test = train_test_split(
                    x_data, x_metadata, y_label, test_size=0.2, random_state=seed, stratify=y_label
                )

            x_data_train, x_data_vad, x_metadata_train, x_metadata_vad, y_label_train, y_label_vad = train_test_split(
                    _x_train, _x_metadata_train, _y_train, test_size=0.2, 
                    random_state=seed, stratify=_y_train
                )
    
    elif dataset == "kauh":
        y_set = np.load(feature_dir + "train_test_split.npy")
        if label in ["copd", "asthma"]:
            mask = (y_label == 0) | (y_label == 1)
            y_label = y_label[mask]
            y_set = y_set[mask]
            x_data = x_data[mask]
            x_metadata = x_metadata[mask]
        x_data_train, x_data_test, _, _ = train_test_split_from_list(x_data, y_label, y_set)
        x_metadata_train, x_metadata_test, y_label_train, y_label_test = train_test_split_from_list(x_metadata, y_label, y_set)
        x_data_train, x_data_vad, x_metadata_train, x_metadata_vad, y_label_train, y_label_vad = train_test_split(
                x_data_train, x_metadata_train, y_label_train, test_size=0.1, 
                random_state=1337, stratify=y_label_train
            )
    elif dataset == "icbhidisease":
        # y_set = np.load(feature_dir + "split.npy")
        y_set = np.load(feature_dir + "segmented_split.npy")
        mask = (y_label == "Healthy") | (y_label == "COPD")
        y_label = y_label[mask]
        y_set = y_set[mask]
        x_data = x_data[mask]
        label_dict = {"Healthy": 0, "COPD": 1}
        y_label = np.array([label_dict[y] for y in y_label])

        x_data_train, x_data_test, y_label_train, y_label_test = train_test_split_from_list(x_data, y_label, y_set)
        x_metadata_train, x_metadata_test, y_label_train, y_label_test = train_test_split_from_list(x_metadata, y_label, y_set)

        x_data_train, x_data_vad, x_metadata_train, x_metadata_vad, y_label_train, y_label_vad = train_test_split(
                x_data_train, x_metadata_train, y_label_train, test_size=0.2, 
                random_state=1337, stratify=y_label_train
            )

    else:
        if dataset == "coviduk":
            y_set = np.load(feature_dir + "split_{}.npy".format(modality))
        elif dataset == "copd":
            y_set = np.load(feature_dir + "train_test_split.npy")
        elif dataset == "coughvid":
            y_set = np.load(feature_dir + "split_{}.npy".format(label))
        x_data_train = x_data[y_set == "train"]
        y_label_train = y_label[y_set == "train"]
        x_metadata_train = x_metadata[y_set == "train"]

        x_data_vad = x_data[y_set == "val"]
        y_label_vad = y_label[y_set == "val"]
        x_metadata_vad = x_metadata[y_set == "val"]

        x_data_test = x_data[y_set == "test"]
        y_label_test = y_label[y_set == "test"]
        x_metadata_test = x_metadata[y_set == "test"]
        # split = np.load(feature_dir + "split.npy")
    
    if task in ["4.5", "4.6"]:
        x_data_train, x_metadata_train, y_label_train = downsample_balanced_dataset(x_data_train, x_metadata_train, y_label_train)
    if task in ["7"]:
        x_data_train, x_metadata_train, y_label_train = upsample_balanced_dataset(x_data_train, x_metadata_train, y_label_train)

    train_data_percentage = configs.train_pct
    if not sample and train_data_percentage < 1:
        x_data_train, _, y_label_train, _, x_metadata_train, _ = train_test_split(
            x_data_train, y_label_train, x_metadata_train, test_size=1 - train_data_percentage, random_state=seed, stratify=y_label_train
        )

    print(collections.Counter(y_label_train))
    min_train_cls = min(collections.Counter(y_label_train).values())
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))
    min_test_cls = min(collections.Counter(y_label_test).values())

    if sample:
        sample_info = [configs.meta_val_iter, n_cls[task], configs.meta_val_shot, configs.meta_val_query]
        if min_train_cls < configs.meta_val_shot:
            sample_info[2] = min_train_cls
        if min_test_cls < configs.meta_val_query:
            sample_info[3] = min_test_cls
        
        if dataset == "coswara" and label =="covid":
            sample_info = [configs.meta_val_iter, n_cls[task] * 2, configs.meta_val_shot // 2, configs.meta_val_query]
            sampler = TrainCategoriesSampler(group_idxs, *sample_info)
        else:
            sampler = TrainCategoriesSampler(y_label_train, *sample_info)
        
        train_data = AudioDataset((x_data_train, y_label_train),  from_audio=from_audio)
        test_data = AudioDataset((x_data_test, y_label_test),  from_audio=from_audio)
        val_data = AudioDataset((x_data_vad, y_label_vad),  from_audio=from_audio)
        
        train_loader = DataLoader(
            train_data, num_workers=2,  batch_sampler=sampler,
        )
        val_loader = DataLoader(
            val_data, num_workers=2, # batch_sampler=sampler,
        )
        test_loader = DataLoader(
            test_data, batch_size=configs.batch_size, shuffle=False, num_workers=2
        )
        return train_loader, val_loader, test_loader
    else:
        train_data = AudioDataset((x_data_train, y_label_train),  from_audio=from_audio)
        test_data = AudioDataset((x_data_test, y_label_test),  from_audio=from_audio)
        val_data = AudioDataset((x_data_vad, y_label_vad),  from_audio=from_audio)
        train_loader = DataLoader(
            train_data, batch_size=configs.batch_size, num_workers=2, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=configs.batch_size, num_workers=2, shuffle=True
        )
        test_loader = DataLoader(
            test_data, batch_size=configs.batch_size, shuffle=True, num_workers=2
        )
        return train_loader, val_loader, test_loader


def test(model, test_loader, loss_func, n_cls, return_auc=False, verbose=True):
    total_loss = []
    test_step_outputs = []
    features = []
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    with torch.no_grad():
        for i,  (x1,y) in enumerate(test_loader):
            x1 = x1.to(device)
            y = y.to(device)
            # print(n_cls, y)
            y_hat = model(x1)

            loss = loss_func(y_hat, y)
            total_loss.append(loss.item())

            _, predicted = torch.max(y_hat, 1)
            probabilities = F.softmax(y_hat, dim=1)
            test_step_outputs.append((y.detach().cpu().numpy(), predicted.detach().cpu().numpy(), probabilities.detach().cpu().numpy() ))
    
    all_outputs = test_step_outputs
    y = np.concatenate([output[0] for output in all_outputs])
    
    total_loss = np.average(total_loss)
    print("loss", total_loss)

    predicted = np.concatenate([output[1] for output in all_outputs])
    probs = np.concatenate([output[2] for output in all_outputs])

    acc = np.mean(predicted == y)

    auroc = AUROC(task="multiclass", num_classes=n_cls)
    auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))
    if verbose:
        print("loss", total_loss)
        print("acc", acc)
        print("auc", auc)
    if return_auc:
        return acc, auc
    return total_loss / (i + 1)


def train_ablation(configs):
    train_loaders, val_loaders , test_loaders = [], [], []
    num_batch = []

    meta_train_loaders, meta_test_loaders = [], []

    for task in configs.train_tasks:
        train_loader, val_loader, _ = get_dataloader(configs, task)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    
    for task in configs.test_tasks:
        _, _, test_loader = get_dataloader(configs, task)
        test_loaders.append(test_loader)
        if configs.few_shot:
            # _, _, meta_test_loader = get_dataloader(configs, task, sample=True)
            meta_train_loader, _, meta_test_loader = get_dataloader(configs, task, sample=True)
            meta_train_loaders.append(meta_train_loader)
            meta_test_loaders.append(meta_test_loader)

    n_cls = collections.defaultdict(lambda:2, {"11": 5, "12": 5})
    # train_loader = combine_dataloaders(train_loaders, train=True)
    # val_loader = combine_dataloaders(val_loaders)

    time_now = time.time()
    train_steps = len(train_loader)
    pretrained_model = initialize_pretrained_model(configs.audio_encoder)
    encoder_path = get_encoder_path(configs.audio_encoder)
    print("loading weights from", encoder_path)
    ckpt = torch.load(encoder_path)
    pretrained_model.load_state_dict(ckpt["state_dict"], strict=False)

    net = pretrained_model.encoder
    model = AudioClassifier(net=net, head=configs.head, classes=configs.n_cls, lr=configs.lr, l2_strength=configs.l2_strength, feat_dim=configs.enc_dim, freeze_encoder=configs.freeze_encoder)
    model = model.to(DEVICE)

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
        train_step_outputs = []

        i = 0
        while True:
            i += 1
            # Randomly select a dataloader
            # selected_idx = random.randint(0, len(configs.train_tasks) - 1)
            selected_idx = random.choices(range(len(configs.train_tasks)), weights=num_batch, k=1)[0]
            selected_iterator = iterators[selected_idx]
            
            try:
                # Get the next batch from the selected dataloader
                x1, y = next(selected_iterator)
            except StopIteration:
                # If any iterator is exhausted, break the loop
                break
            
            x1 = x1.to(DEVICE)
            y = y.to(DEVICE)
            iter_count += 1
            model_optim.zero_grad()

            y_hat = model(x1)
            loss = loss_func(y_hat, y)
            train_loss.append(loss.item())

            _, predicted = torch.max(y_hat, 1)

            probabilities = F.softmax(y_hat, dim=1)
            train_step_outputs.append((y.detach().cpu().numpy(), predicted.detach().cpu().numpy(), probabilities.detach().cpu().numpy() ))
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
            test(model, train_loader, loss_func, configs.n_cls)
        
        print("="*10 + "validation")
        validation_loss = 0
        for j, val_loader in enumerate(val_loaders):
            validation_loss += test(model, val_loader, loss_func, configs.n_cls)

        print("="*10 + "test")
        for j, test_loader in enumerate(test_loaders):
            # test
            print("Task", configs.test_tasks[j])
            if  n_cls[configs.test_tasks[j]] != configs.n_cls:
                pass
            else:
                test(model, test_loader, loss_func, configs.n_cls)
        
        if early_stopper.early_stop(validation_loss):
            print("early stopping")      
            break
    

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_encoder", type=str, default="operaCT")
    parser.add_argument("--train_tasks", type=str, default="1,2,4.1,4.2,4.5,4.6,7")
    parser.add_argument("--test_tasks", type=str, default="1,2,4.1,4.2,4.5,4.6,7,8,8.1,8.2,8.3,8.5,8.6,8.7,8.8,10,10.5,10.6")
    parser.add_argument("--enc_dim", type=int, default=768)

    parser.add_argument("--n_run", type=int, default=5)
    parser.add_argument("--head", type=str, default="linear")
    parser.add_argument("--freeze_encoder", type=str, default='none') # "none" to finetune encoder

    parser.add_argument("--n_run_finetune", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_pct", type=float, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2_strength", type=float, default=1e-5) 
    parser.add_argument("--n_cls", type=int, default=2)

    parser.add_argument("--few_shot", default=True)

    parser.add_argument("--meta-val-interval", type=int, default=4) #4
    parser.add_argument("--meta_val_iter", type=int, default=10)
    parser.add_argument("--meta_val_way", type=int, default=5)
    parser.add_argument("--meta_val_shot", type=int, default=20)
    parser.add_argument("--meta_val_query", type=int, default=15)
    parser.add_argument("--meta_val_metric", type=str, default="euclidean") # euclidean, cosine, l1, l2

    parser.add_argument("--few_shot_finetuning", type=bool, default=False)
    parser.add_argument("--data_efficient_finetuning", type=bool, default=False)
    args = parser.parse_args()

    print(args)

    args.train_tasks = args.train_tasks.split(",")
    args.test_tasks = args.test_tasks.split(",")

    train_ablation(args)