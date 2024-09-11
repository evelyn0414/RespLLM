
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
from src.util import train_test_split_from_list, downsample_balanced_dataset, upsample_balanced_dataset,plot_tsne
from src.benchmark.linear_eval import FeatureDataset, DecayLearningRate
import collections
from tqdm import tqdm
from src.benchmark.llm_eval.sampler import SplitCategoriesSampler
from lightning.pytorch.utilities import CombinedLoader
import time
from torch import nn
from torch.nn import functional as F
from src.benchmark.llm_eval.util import itr_merge, EarlyStopper, get_context, downsample_balanced_dataset, upsample_balanced_dataset
from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model
import random
import numbers
from torchmetrics import AUROC
import pandas as pd
import torch.optim as optim
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def weights_init(network):
    for m in network:
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Linear') != -1:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()


class AudioClassifierConcat(pl.LightningModule):
    def __init__(self, net, head="linear", feat_dim=1280, metadata_dim=44, classes=4, lr=1e-4, loss_func=None, freeze_encoder="none", l2_strength=0.0005, use_audio=True):
        super().__init__()
        self.net = net
        self.freeze_encoder = freeze_encoder

        if freeze_encoder == "all":
            for param in self.net.parameters():
                param.requires_grad = False
        elif freeze_encoder == "early":
            # print(self.net)
            # Selective freezing (fine-tuning only the last few layers), name not matching yet
            for name, param in self.net.named_parameters():
                # print(name)
                if 'cnn1' in name or 'efficientnet._blocks.0.' in name or 'efficientnet._blocks.1.' in name or "efficientnet._blocks.2." in name or "efficientnet._blocks.3." in name or "efficientnet._blocks.4." in name: 
                    # for efficientnet
                    param.requires_grad = True
                    print(name)
                elif 'patch_embed' in name or 'layers.0' in name or 'layers.1' in name or 'layers.2' in name or "htsat.norm" in name or "htsat.head" in name or "htsat.tscam_conv" in name:
                    # for htsat
                    param.requires_grad = True
                    print(name)
                else:
                    param.requires_grad = False
                    # print(name)
        
        self.use_audio = use_audio

        if head == 'linear':
            if self.use_audio:
                self.head = nn.Sequential(
                    nn.BatchNorm1d(feat_dim + metadata_dim),
                    nn.Linear(feat_dim + metadata_dim, classes)
                                          )
            else:
                self.head = nn.Sequential(nn.Linear(metadata_dim, classes))
        elif head == 'mlp':
            if self.use_audio:
                self.head = nn.Sequential(
                    nn.Linear(feat_dim + metadata_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(128, classes)
                )
            else:
                self.head = nn.Sequential(
                    nn.Linear(metadata_dim, metadata_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(metadata_dim, classes)
                )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
        
        weights_init(self.head)
        self.lr = lr
        # self.l2_strength = l2_strength
        self.l2_strength_new_layers = l2_strength
        self.l2_strength_encoder = l2_strength * 0.2
        self.loss = loss_func if loss_func else nn.CrossEntropyLoss()
        self.classes = classes
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x, c):
        if self.use_audio:
            x = self.net(x)
            # print(x.shape, c.shape)
            x = torch.cat([x, c], dim=1)
        else:
            x = c
        # print(x.shape)
        return self.head(x)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, from_npy=False,  from_audio=False):
        self.data = data[0]
        self.metadata = data[1]
        self.label = data[2]
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
        metadata =  self.metadata[idx]

        if self.from_audio:
            return x, label

        x = torch.tensor(x, dtype=torch.float)
        metadata = torch.tensor(metadata, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return x, metadata, label


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


    # process tensor embedding of the metadata
    metadata_npy_filename = feature_dir + f"metadata_embedding_{configs.context_encoding}" + suffix_dataset
    if not os.path.exists(metadata_npy_filename):

        if dataset == "coswara":
            sound_dir_loc = np.load(
            feature_dir + "{}_aligned_filenames_{}_w_{}.npy".format(broad_modality, label, modality))
        elif dataset == "kauh":
            sound_dir_loc = np.load(feature_dir + "sound_dir_loc_subset.npy")
        else:
            sound_dir_loc = np.load(feature_dir + "sound_dir_loc" + suffix_dataset)
        
        if dataset == "ssbpr":
            gender = [filename.split("/")[-3] for filename in sound_dir_loc]
            metadata = [{'gender': gender[i]} for i in range(len(gender))]
        elif dataset == "covid19sounds":
            if task in ["4.1", "4.2"]:
                import glob as gb
                metadata = []
                data_df = pd.read_csv("datasets/covid19-sounds/data_0426_en_task2.csv")
                metadata_dir = np.array(gb.glob("datasets/covid19-sounds/covid19_data_0426_metadata/*.csv"))
                df = None
                # use metadata as outer loop to enable quality check
                for file in metadata_dir:
                    if df is None:
                        df = pd.read_csv(file, delimiter=";")
                    else:
                        df = pd.concat([df, pd.read_csv(file, delimiter=";")])
                for _, data_row in data_df.iterrows():
                    try:
                        # print(data_row)
                        uid = data_row["cough_path"].split("\\")[0]
                        # print(uid)
                        folder_name = data_row["cough_path"].split("\\")[1]
                        # print(folder_name)
                        row = df[df['Uid'] == uid]
                        row = row[row['Folder Name'] == folder_name]
                        row = row.iloc[0]
                        # print(row)
                        metadata.append({'age': row["Age"], 
                                    "gender": row["Sex"], 
                                    "medhistory": row["Medhistory"], 
                                    "symptoms": row["Symptoms"]
                                    } )
                    except IndexError:
                        print("metadata nonexist", uid, folder_name)
                        metadata.append({} )
            elif task in ["4.5", "4.6"]:
                df = pd.read_csv("datasets/covid19-sounds/data_0426_en_task1.csv", delimiter=";", index_col="Uid")
                # print(df.head(5))
                df = df[~df.index.duplicated(keep='first')]
                metadata = []
                for filename in sound_dir_loc:
                    uid = filename.split("/")[-3]
                    if uid == "form-app-users":
                        uid = filename.split("/")[-2]
                    row = df.loc[uid]
                    # print(row)
                    metadata.append({'age': row["Age"], 
                                    "gender": row["Sex"], 
                                    "medhistory": row["Medhistory"], 
                                        "symptoms": row["Symptoms"]
                                    } )
            else:
                raise NotImplementedError(f"task {task} not implemented")
        elif dataset == "coviduk":
            participant_data = pd.read_csv("datasets/covidUK/audio_metadata.csv")
            split_data = pd.read_csv("datasets/covidUK/participant_metadata.csv")
            df = pd.merge(participant_data, split_data, on='participant_identifier')
            metadata = []
            # df.fillna(0)
            df = df.replace(np.nan, 0, regex=True)
            for filename in sound_dir_loc:
                audio_name = filename.split("/")[-1]
                row = df.loc[df[f"{modality}_file_name"] == audio_name]
                # print(row)
                # print(int(row["respiratory_condition_asthma"]))
                medhistory = ",".join([med for med in ["respiratory_condition_asthma", "respiratory_condition_other"] if int(row[med])])

                symptoms = ""
                if int(row["symptom_none"]):
                    symptoms = "None"
                elif int(row["symptom_prefer_not_to_say"]):
                    symptoms = "pnts"
                else:
                    syms = ["cough_any", "new_continuous_cough", "runny_or_blocked_nose", "shortness_of_breath", "sore_throat", "abdominal_pain", "diarrhoea", "fatigue", "fever_high_temperature", "headache", "change_to_sense_of_smell_or_taste", "loss_of_taste"]
                    symptoms = ",".join([sym for sym in syms if int(row["symptom_" + sym])])
                    # print(symptoms)
                
                metadata.append({'age': row["age"].values[0], 
                                "gender": row["gender"].values[0], 
                                "smoke status": row["smoker_status"].values[0],
                                "medhistory": medhistory, 
                                    "symptoms": symptoms
                                } )
        elif dataset == "copd":
            location = [filename.split("/")[-1][5:7] for filename in sound_dir_loc]
            metadata = [{"location": location[i]} for i in range(len(location))]
        elif dataset == "kauh":
            metadata = []
            for filename in sound_dir_loc:
                location = filename.split(",")[-3]
                gender = filename.split(",")[-1].split(".")[0]
                metadata.append({
                    "location": location,
                    "gender": gender
                })
        elif dataset == "icbhidisease":
            metadata = []
            df = pd.read_csv('datasets/icbhi/ICBHI_Challenge_demographic_information.txt',
                                dtype=str, sep='\t', names=['userId', 'Age', 'Sex', 'Adult_BMI', 'Child Weight', 'Child Height'],  index_col="userId")
            for filename in sound_dir_loc:
                userID = int(filename.split("/")[-1].split("_")[0])
                location = filename.split("/")[-1].split("_")[2]
                row = df.loc[userID]
                metadata.append({
                    "age": row["Age"],
                    "gender": row["Sex"],
                    "location": location
                })
        elif dataset == "coswara":
            df = pd.read_csv("datasets/Coswara-Data/combined_data.csv", index_col="id")
            metadata = []
            for filename in sound_dir_loc:
                uid = filename.split("/")[-2]
                row = df.loc[uid]
                # print(row)
                syms = ["cold", "cough", "fever", "diarrhoea", "st", "loss_of_smell", "mp", "ftg", "bd"]
                symptoms = ",".join([sym for sym in syms if row[sym] is True])
                # print(symptoms)
                metadata.append({
                    "age": row["a"],
                    "gender": row["g"],
                    "symptoms": symptoms,
                    # TODO : vacc?
                })
        elif dataset == "coughvid":
            df = pd.read_csv("datasets/coughvid/metadata_compiled.csv", index_col="uuid")
            metadata = []
            for filename in sound_dir_loc:
                uid = filename.split("/")[-1][:-4]
                
                try:
                    row = df.loc[uid]
                    metadata.append({
                        "age": row["age"],
                        "gender": row["gender"],
                        "symptoms": "fever_muscle_pain" if row["fever_muscle_pain"] else "",
                    })
                except KeyError:
                    print(uid, "not found")
                    metadata.append({})
        else:
            print("metadata not included for", dataset)
            metadata = np.array([{} for x in range(len(sound_dir_loc))])
        # for key in metadata[0]:
        #     print(key, collections.Counter([data[key] for data in metadata]))

        if configs.context_encoding == "dense":
            # process dense embedding
            x_metadata = np.array([get_dense(d) for d in metadata])
        elif configs.context_encoding == "wordemb":
            x_metadata = np.array(get_llama_embedding(metadata))
        elif configs.context_encoding == "bertemb":
            x_metadata = np.array(get_bert_embedding(metadata))
        else:
            raise NotImplementedError(f"unknown context encoding {configs.context_encoding}")

        np.save(metadata_npy_filename, x_metadata)
    
    x_metadata = np.load(metadata_npy_filename)
    # print(x_metadata[11])
    
    from_audio = False
    seed = 42
    if dataset == "icbhidisease":
        x_data = np.load(feature_dir + f"segmented_spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset)
        y_label = np.load(feature_dir + f"segmented_labels.npy")
        index_sampled = np.load(feature_dir + f"segmented_index.npy")
        x_metadata = x_metadata[index_sampled]
    else:
        x_data = np.load(feature_dir + f"spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset)
        
    print(len(x_data), len(y_label))
    print(collections.Counter(y_label))


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
        
            for i in range(len(x_data_test)):
                y = y_label_test[i]
                m = symptoms_test[i]
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
        x_metadata = x_metadata[mask]
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

        all_data = AudioDataset((np.concatenate((x_data_train, x_data_test)), np.concatenate((x_metadata_train,x_metadata_test)), np.concatenate((y_label_train, y_label_test))),  from_audio=from_audio)
        # ["val"] * len(y_label_vad) + 
        split = ["train"] * len(y_label_train) + ["test"] * len(y_label_test)
        
        if dataset == "coswara" and label =="covid":
            sample_info = [configs.meta_val_iter, n_cls[task] * 2, configs.meta_val_shot // 2, configs.meta_val_query]
            sampler = SplitCategoriesSampler(np.array(group_idxs), split, *sample_info)
        else:
            sampler = SplitCategoriesSampler(np.concatenate((y_label_train, y_label_test)), split, *sample_info)
        fewshot_loader = DataLoader(
            all_data, num_workers=2, batch_sampler=sampler,
        )
        return fewshot_loader
    else:
        train_data = AudioDataset((x_data_train, x_metadata_train, y_label_train),  from_audio=from_audio)
        test_data = AudioDataset((x_data_test, x_metadata_test, y_label_test),  from_audio=from_audio)
        val_data = AudioDataset((x_data_vad, x_metadata_vad, y_label_vad),  from_audio=from_audio)
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


def get_dense(metadata):
    # print(metadata)
    emb = []

    age_mapping = {'30-39': 3, '40-49': 4, '20-29': 2, '50-59': 5, '60-69': 6, '16-19': 1, '70-79': 7, 'pnts': 0, '0-19': 1, '80-89': 8, 'Prefer not to say': 0, 'missing': 0, '90-': 9}
    if "age" in metadata:
        age = metadata["age"]
        if age in age_mapping:
            emb.append(age_mapping[age])
        else:
            # try to convert to the most similar category
            if isinstance(age, numbers.Number):
                try:
                    emb.append(int(age) // 10)
                except ValueError:
                    # NaN
                    emb.append(0)
            elif age[0].isdigit():
                emb.append(int(age[0]))
            else:
                emb.append(0)
    else:
        emb.append(0)

    gender_mapping = collections.defaultdict(lambda:0, {"female": 1, "male": 2, "f":1, "m":2, 'Female': 1, 'Male': 2, "F":1, "M":2})
    if "gender" in metadata:
        emb.append(gender_mapping[metadata["gender"]])
    else:
        emb.append(0)

    location_mapping = collections.defaultdict(lambda:0, {
        'Al': 1,
        'Ar': 2, 
        'Pl': 3, 
        'Pr': 4, 
        'Ll': 5, 
        'Lr': 6,
        'Tc': 7,
        })
    if "location" in metadata:
        emb.append(location_mapping[metadata["location"]])
    else:
        emb.append(0)
    
    med_history_mapping = { 
            'angina':"Angina",
            'asthma': "Asthma",
            'cancer':"Cancer",
            'copd': "COPD/Emphysema",
            'cystic': "Cystic fibrosis", 
            'diabetes': "Diabetes", 
            'hbp': "High Blood Pressure", 
            'heart': "Previous heart attack",
            'hiv': "HIV or impaired immune system",
            'long': "Other long-term condition",
            'longterm': "Other long-term condition",
            'lung':"Other lung disease",
            'otherHeart': "Other heart disease",
            'organ': "Previous organ transplant",
            'pulmonary':"Pulmonary fibrosis", 
            'stroke': "Previous stroke or Transient ischaemic attack", 
            'valvular': "Valvular heart disease"
            }

    if "medhistory" not in metadata:
        emb.extend([0] * len(med_history_mapping))
    else:
        userMedHistory = metadata["medhistory"]
        if pd.isna(userMedHistory) or userMedHistory == "" or "None" in userMedHistory or "none" in userMedHistory:
        # no medhistory mapped
            emb.extend([0] * len(med_history_mapping))
        else:
            userMedHistory = metadata["medhistory"]
            if userMedHistory[0] == ",": userMedHistory = userMedHistory[1:]
            if userMedHistory[-1] == ",": userMedHistory = userMedHistory[:-1]
            userMedHistory = userMedHistory.split(",")
            for med in med_history_mapping:
                emb.append(1 if med in userMedHistory else 0)
    
    symptoms_mapping = { 
        'drycough': "Dry cough", 
        'wetcough': "Wet cough", 
        'sorethroat': "Sore throat", 
        'runnyblockednose': "Runny or blocked nose",
        'runny': "Runny or blocked nose",
        'tightness': "Tightness in the chest", 
        'smelltasteloss': "Loss of taste and smell", 
        'fever': "Fever", 
        'chills': "Chills",  
        'shortbreath': "Difficulty breathing or feeling short of breath", 
        'dizziness': "Dizziness, confusion or vertigo", 
        'headache': "Headache", 
        'muscleache': "Muscle aches",
        # covid uk
        "cough_any": "cough",
        "new_continuous_cough": "a new continuous cough", 
        "runny_or_blocked_nose": "runny or blocked nose", 
        "shortness_of_breath": "shortness of breath", 
        "sore_throat": "sore throat", 
        "abdominal_pain": "abdominal pain", 
        "diarrhoea": "diarrhoea", 
        "fatigue": "fatigue", 
        "fever_high_temperature": "fever or a high temperature", 
        "headache": "headache", 
        "change_to_sense_of_smell_or_taste": "a change to sense of smell or taste", 
        "loss_of_taste": "loss of sense of taste"
        }
    
    if "symptoms" not in metadata:
        emb.extend([0] * len(symptoms_mapping))
    else:
        userSymptoms = metadata["symptoms"]
        if pd.isna(userSymptoms) or userSymptoms == "" or "None" in userSymptoms or "pnts" in userSymptoms:
            # print(row)
            emb.extend([0] * len(symptoms_mapping))
        else:
            if userSymptoms[0] == ",": userSymptoms = userSymptoms[1:]

            userSymptoms = userSymptoms.split(",")
            for sym in symptoms_mapping:
                emb.append(1 if sym in userSymptoms else 0)

    # print(len(emb))
    return np.array(emb)


def get_bert_embedding(metadata):
    from transformers import BertModel, BertTokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    out = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    for m in tqdm(metadata):
        sentence = get_context(m)

        # Tokenize the input sentence
        inputs = tokenizer(sentence, return_tensors="pt")

        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Get the model's output
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the embeddings for all tokens in the sentence
        last_hidden_state = outputs.last_hidden_state

        # Option 1: Use the [CLS] token embedding (if the model has a [CLS] token)
        # sentence_embedding = last_hidden_state[:, 0, :]

        # Option 2: Mean pooling of all token embeddings
        sentence_embedding = torch.mean(last_hidden_state, dim=1)

        sentence_embedding = torch.squeeze(sentence_embedding, dim=0)

        # print(sentence_embedding.shape)

        out.append(sentence_embedding.cpu().numpy())

    return np.array(out)


def test(model, test_loader, loss_func, n_cls):
    total_loss = []
    test_step_outputs = []
    features = []
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    with torch.no_grad():
        for i,  (x1, x2,y) in enumerate(test_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            # print(n_cls, y)
            y_hat = model(x1, x2)
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
    print("acc", acc)
    print("auc", auc)
    return total_loss / (i + 1)


def few_shot_finetune_linear(test_loader, model, configs, sample_group=False):
    original_linear_state_dict = model.head.state_dict()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    # Define optimizer, only for the linear head
    optimizer = optim.Adam(model.head.parameters(), lr=1e-3)

    # Assuming you have a loss function and data loader for dataset B
    loss_fn = nn.CrossEntropyLoss()

    meta_val_query = test_loader.batch_sampler.n_query
    if sample_group:
        meta_val_way = test_loader.batch_sampler.n_way // 2
        meta_val_shot = test_loader.batch_sampler.n_shot * 2
    else:
        meta_val_way = test_loader.batch_sampler.n_way
        meta_val_shot = test_loader.batch_sampler.n_shot
    
    # print(f"few shot linear evaluation: n_way:{meta_val_way}, n_shot:{meta_val_shot}, n_query:{meta_val_query}.")
    # Fine-tuning loop (simplified)
    probs, y = [], []
    test_step_outputs = []
    for i, (x1, x2, target) in enumerate(test_loader):
        # print(x1.shape, x2.shape, target.shape)
        x1 = x1.to("cuda")
        x2 = x2.to("cuda")
        target = target.cuda(0, non_blocking=True)
        
        x1_train = x1[:meta_val_way * meta_val_shot]
        x1_test = x1[meta_val_way * meta_val_shot:]
        x2_train = x2[:meta_val_way * meta_val_shot]
        x2_test = x2[meta_val_way * meta_val_shot:]
        
        if i == 0 or meta_val_query > 0:
            model.train()
            for epoch in range(3):
                optimizer.zero_grad()
                outputs = model(x1_train, x2_train)
                loss = loss_fn(outputs, target[:meta_val_way * meta_val_shot])
                loss.backward()
                optimizer.step()
        model.eval()
        y_hat = model(x1_test, x2_test)
        _, predicted = torch.max(y_hat, 1)
        probabilities = F.softmax(y_hat, dim=1)
        y_true = target[meta_val_way * meta_val_shot:]

        test_step_outputs.append((y_true.cpu().numpy(), predicted.detach().cpu().numpy(), probabilities.detach().cpu().numpy() ))
        if meta_val_query > 0:
            model.head.load_state_dict(original_linear_state_dict)
    
    model.head.load_state_dict(original_linear_state_dict)
    all_outputs = test_step_outputs
    y = np.concatenate([output[0] for output in all_outputs])
    predicted = np.concatenate([output[1] for output in all_outputs])
    probs = np.concatenate([output[2] for output in all_outputs])

    acc = np.mean(predicted == y)

    auroc = AUROC(task="multiclass", num_classes=configs.n_cls)
    print("test_acc", acc)
    auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))
    print("test_auc", auc)
    return acc, auc


def data_efficient_finetuning(task, model, train_pct, args, mode="linear", seed=None):
    """
    This function changes the requires_grad value of parameters and does not restore.
    Only use it for final evaluation end of training.
    """
    import argparse, copy

    n_cls = 5 if task in ["11", "12"] else 2
    
    print(f"data efficient finetuning on task {task} with {train_pct} of total data.")
    # configs = args.deepcopy()
    configs = argparse.Namespace(**{k: copy.deepcopy(v) for k, v in vars(args).items()})
    configs.train_pct = train_pct
    train_loader, val_loader, test_loader = get_dataloader(configs, task, deft_seed=seed)

    original_linear_state_dict = model.head.state_dict()
    
    # flag = 0
    # if n_cls != model.n_cls:
    #     model.reinitialize_clf(n_cls)
    #     model = model.to(DEVICE)
    #     flag = 1

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    # Define optimizer, only for the linear head
    model_optim = optim.Adam(model.head.parameters(), lr=1e-4)

    # Assuming you have a loss function and data loader for dataset B
    loss_func = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=1, min_delta=0.01)

    for epoch in tqdm(range(configs.finetune_epochs)):
        iter_count = 0
        train_loss = []

        model.train()
        # train_step_outputs = []

        for i, (x1, x2, y) in enumerate(train_loader):
            x1 = x1.to(DEVICE)
            x2 = x2.to(DEVICE)
            y = y.to(DEVICE)
            iter_count += 1
            model_optim.zero_grad()

            y_hat = model(x1, x2)
            loss = loss_func(y_hat, y)
            train_loss.append(loss.item())

            if i < 3 or (i + 1) % 10 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
        
        loss.backward()
        model_optim.step()

        print("=" * 5 + "val")
        validation_loss = test(model, val_loader, loss_func, n_cls)
        

        if early_stopper.early_stop(validation_loss):
            print("early stopping")      
            break
    
    print("=" * 5 + "test")
    print("Task", task)
    test(model, test_loader, loss_func, n_cls)

    model.head.load_state_dict(original_linear_state_dict)


def train_ablation(configs):
    if configs.context_encoding == "dense":
        metadata_dim = 44
    elif configs.context_encoding == "wordemb":
        metadata_dim = 4096
    elif configs.context_encoding == "bertemb":
        metadata_dim = 768

    train_loaders, val_loaders , test_loaders = [], [], []
    num_batch = []

    meta_test_loaders = []

    for task in configs.train_tasks:
        train_loader, val_loader, _ = get_dataloader(configs, task)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    
    for task in configs.test_tasks:
        _, _, test_loader = get_dataloader(configs, task)
        test_loaders.append(test_loader)
        if configs.few_shot:
            # _, _, meta_test_loader = get_dataloader(configs, task, sample=True)
            meta_test_loader = get_dataloader(configs, task, sample=True)
            meta_test_loaders.append(meta_test_loader)

    n_cls = collections.defaultdict(lambda:2, {"11": 5, "12": 5})

    time_now = time.time()
    train_steps = len(train_loader)
    pretrained_model = initialize_pretrained_model(configs.audio_encoder)
    encoder_path = get_encoder_path(configs.audio_encoder)
    print("loading weights from", encoder_path)
    ckpt = torch.load(encoder_path)
    pretrained_model.load_state_dict(ckpt["state_dict"], strict=False)

    net = pretrained_model.encoder
    model = AudioClassifierConcat(net=net, head=configs.head, classes=configs.n_cls, lr=configs.lr, l2_strength=configs.l2_strength, feat_dim=configs.enc_dim, metadata_dim=metadata_dim, freeze_encoder=configs.freeze_encoder, use_audio=configs.use_audio)
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
                x1, x2, y = next(selected_iterator)
            except StopIteration:
                # If any iterator is exhausted, break the loop
                break
            
            x1 = x1.to(DEVICE)
            x2 = x2.to(DEVICE)
            y = y.to(DEVICE)
            iter_count += 1
            model_optim.zero_grad()

            y_hat = model(x1, x2)
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
        
        if (epoch + 1) % configs.test_interval == 0:
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
    
    print("="*10 + "test")
    for j, test_loader in enumerate(test_loaders):
        # test
        print("Task", configs.test_tasks[j])
        if  n_cls[configs.test_tasks[j]] != configs.n_cls:
            pass
        else:
            test(model, test_loader, loss_func, configs.n_cls)


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
    # parser.add_argument("--task", type=str, default="6to9")
    # parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--enc_dim", type=int, default=768)

    parser.add_argument("--context_encoding", type=str, default="dense") # dense / wordemb / bertemb

    parser.add_argument("--n_run", type=int, default=5)
    parser.add_argument("--head", type=str, default="linear")
    parser.add_argument("--freeze_encoder", type=str, default='none') # "all" for freezing, "none" to finetune encoder

    parser.add_argument("--use_audio", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--n_run_finetune", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_pct", type=float, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2_strength", type=float, default=1e-5) 
    parser.add_argument("--n_cls", type=int, default=2)

    parser.add_argument("--few_shot", default=True)
    parser.add_argument("--test_interval", type=int, default=1) #4
    parser.add_argument("--meta-val-interval", type=int, default=4) #4
    parser.add_argument("--meta_val_iter", type=int, default=10)
    parser.add_argument("--meta_val_way", type=int, default=5)
    parser.add_argument("--meta_val_shot", type=int, default=20)
    parser.add_argument("--meta_val_query", type=int, default=-1)
    parser.add_argument("--meta_val_metric", type=str, default="euclidean") # euclidean, cosine, l1, l2

    args = parser.parse_args()

    print(args)

    args.train_tasks = args.train_tasks.split(",")
    args.test_tasks = args.test_tasks.split(",")

    train_ablation(args)