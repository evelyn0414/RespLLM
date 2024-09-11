import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC
from tqdm import tqdm
import os
import collections
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from src.benchmark.llm_eval.sampler import CategoriesSampler, SplitCategoriesSampler, TrainCategoriesSampler
from src.util import train_test_split_from_list, plot_tsne
import torch.optim as optim
import pickle
import copy
from numpy import linalg as LA
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, from_npy=False, from_audio=False, prompt=""):
        self.data = data[0]
        self.metadata = data[1]
        self.label = data[2]

        # self.max_len = max_len
        # self.augment = augment
        self.from_npy = from_npy
        # self.crop_mode = crop_mode
        self.from_audio = from_audio
        self.prompt = prompt

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

        # if self.max_len:
        #     if self.crop_mode == "random":
        #         x = random_crop(x, crop_size=self.max_len)
        #     else:
        #         x = crop_first(x, crop_size=self.max_len)

        # if self.augment:
        #     x = random_mask(x)
        #     x = random_multiply(x)

        x = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return x, self.prompt, metadata, label


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v

def merge_dataloader(dataloaders):
    return ConcatDataset(dataloaders)

def get_dataloader(configs, task, sample=False, deft_seed=None):
    tasks_config = {
        "1": ("coviduk", "covid", "exhalation"),
        "2": ("coviduk", "covid", "cough"),
        "3": ("covid19sounds", "symptom", "breath"),
        "4": ("covid19sounds", "symptom", "cough"),
        "4.1": ("covid19sounds", "covid", "breath"),
        "4.2": ("covid19sounds", "covid", "cough"),
        "4.5": ("covid19sounds", "smoker", "breath"),
        "4.6": ("covid19sounds", "smoker", "cough"),
        "5": ("coughvid", "covid", "cough"),
        "6": ("coughvid", "gender", "cough"),
        "7": ("icbhidisease", "copd", "lung"),
        "7.5": ("icbhi", "event", "lung"),
        "8": ("coswara", "smoker", "cough-shallow"),
        "8.1": ("coswara", "smoker", "cough-heavy"),
        "8.2": ("coswara", "smoker", "breathing-shallow"),
        "8.3": ("coswara", "smoker", "breathing-deep"),
        "8.5": ("coswara", "covid", "cough-shallow"),
        "8.6": ("coswara", "covid", "cough-heavy"),
        "8.7": ("coswara", "covid", "breathing-shallow"),
        "8.8": ("coswara", "covid", "breathing-deep"),

        "9": ("coswara", "sex", "cough-shallow"),
        "10": ("kauh", "obstructive", "lung"),
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

    # prompt
    prompt = get_prompt(configs, dataset=dataset, label=label, modality=modality)
    print(prompt)

    # metadata
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
    
    if dataset == "coswara":
        sound_dir_loc = np.load(
        feature_dir + "{}_aligned_filenames_{}_w_{}.npy".format(broad_modality, label, modality))
    elif dataset == "kauh":
        sound_dir_loc = np.load(feature_dir + "sound_dir_loc_subset.npy")
    else:
        sound_dir_loc = np.load(feature_dir + "sound_dir_loc" + suffix_dataset)

    if configs.use_context:
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
                    # print(len(df))
                        # df.append(pd.read_csv(file, delimiter=";"))
                    
                    # if "cough" in modality:
                    #     df = df[df["Cough check"].str.contains("c")]
                    # if "breath" in modality:
                    #     df = df[df["Breath check"].str.contains("b")]
                    # if "voice" in modality:
                    #     df = df[df["Voice check"].str.contains("v")]
                # df = df.set_index(["Uid", "Folder Name"])
                # print(df.head(10))
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
                    # symptoms = df.at[(uid, folder_name), "Symptoms"]
                    # metadata.append({'age': df.at[(uid, folder_name), "Age"],
                    #             "gender": df.at[(uid, folder_name), "Sex"],
                    #             "medhistory": df.at[(uid, folder_name), "Medhistory"], 
                    #             "symptoms": df.at[(uid, folder_name), "Symptoms"]
                    #             } )
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
        x_metadata = np.array([get_context(d) for d in metadata])
    
    else:
        x_metadata = np.array(["" for x in range(len(sound_dir_loc))])

    for sample in [0, 11, 12, 13, 24, 34, 666, 717, 1024][:]:
        if sample < len(y_label):
            print("sound_dir_loc", sound_dir_loc[sample])
            print(x_metadata[sample])
            print("y_label", y_label[sample])
    
    # audio data
    from_audio = False

    spec_file_name = feature_dir + f"spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset if dataset != "icbhidisease" else feature_dir + f"segmented_spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset
    if not os.path.exists(spec_file_name):
        from src.util import get_split_signal_librosa
        x_data = []
        if dataset == "icbhidisease":
            y_segmented, y_set_segmented = [], []
            # x_metadata_segmented = []
            index_segmented = []
            y_set = np.load(feature_dir + "split.npy")
            for idx, audio_file in enumerate(sound_dir_loc):
                data = get_split_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=pad_len_htsat[dataset], trim_tail=False)
                if y_set[idx] == "train":
                    # print([y_set[idx]], len(data))
                    x_data.extend(data)
                    y_segmented.extend([y_label[idx]] * len(data))
                    y_set_segmented.extend([y_set[idx]] * len(data))
                    # x_metadata_segmented.extend([x_metadata[idx]] * len(data))
                    index_segmented.extend([idx] * len(data))
                else:
                    # print([y_set[idx]])
                    x_data.append(data[0])
                    y_segmented.append(y_label[idx])
                    y_set_segmented.append(y_set[idx])
                    # x_metadata_segmented.append([x_metadata[idx]])
                    index_segmented.append(idx)
            x_data = np.array(x_data)
            y_segmented = np.array(y_segmented)
            y_set_segmented = np.array(y_set_segmented)
            np.save(spec_file_name, x_data)
            np.save(feature_dir + f"segmented_split.npy", y_set_segmented)
            np.save(feature_dir + f"segmented_labels.npy", y_segmented)
            np.save(feature_dir + f"segmented_index.npy", index_segmented)
        else:
            for audio_file in sound_dir_loc:
                data = get_split_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=pad_len_htsat[dataset], trim_tail=False)[0]
                # print(data.shape)
                x_data.append(data)
            x_data = np.array(x_data)
            np.save(spec_file_name, x_data)

    seed = 42

    if dataset == "icbhidisease":
        x_data = np.load(feature_dir + f"segmented_spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset)
        y_label = np.load(feature_dir + f"segmented_labels.npy")
        index_sampled = np.load(feature_dir + f"segmented_index.npy")
        x_metadata = x_metadata[index_sampled]
    else:
        x_data = np.load(feature_dir + f"spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset)
    print(len(x_data), len(x_metadata), len(y_label))
    
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
        if True: #label == "covid":
            set_all_seed(seed)
            symptoms = np.array([1 if 'following respiratory symptoms' in m else 0 for m in x_metadata])
            np.save(feature_dir + f"symptom" + suffix_dataset, symptoms)
            # symptoms = np.load(feature_dir + f"symptom" + suffix_dataset)

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

            x_data_train = x_data[indices_train]
            x_metadata_train = x_metadata[indices_train]
            y_label_train = y_label[indices_train]

            x_data_vad, x_metadata_vad, y_label_vad = x_data_train, x_metadata_train, y_label_train 

            group_idxs = []
            for i in range(len(x_data_train)):
                y = y_label_train[i]
                m = x_metadata_train[i]
                if y == 0 and 'following respiratory symptoms' in m:
                    group = 1
                if y == 0 and 'following respiratory symptoms' not in m:
                    group = 2
                if y == 1 and 'following respiratory symptoms' in m:
                    group = 3
                if y == 1 and 'following respiratory symptoms' not in m:
                    group = 4
                group_idxs.append(group)
        
            group_idxs = np.array(group_idxs)

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


    if task in ["4.5", "4.6", "10.6"]:
        x_data_train, x_metadata_train, y_label_train = downsample_balanced_dataset(x_data_train, x_metadata_train, y_label_train)
    if task in ["7"]:
        x_data_train, x_metadata_train, y_label_train = upsample_balanced_dataset(x_data_train, x_metadata_train, y_label_train)
    if task in ["10.6"]:
        x_data_train, x_metadata_train, y_label_train = downsample_balanced_dataset(x_data_train, x_metadata_train, y_label_train)
        x_data_test, x_metadata_test, y_label_test = downsample_balanced_dataset(x_data_test, x_metadata_test, y_label_test)

        x_data_train, x_metadata_train, y_label_train, x_data_test, x_metadata_test, y_label_test = x_data_test, x_metadata_test, y_label_test, x_data_train, x_metadata_train, y_label_train


    # !! didn't split the metadata as needed, all results were wrong
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

    # print(x_metadata_test)

    if sample and configs.few_shot:
        sample_info = [configs.meta_val_iter, n_cls[task], configs.meta_val_shot, configs.meta_val_query]
        if min_train_cls < configs.meta_val_shot:
            sample_info[2] = min_train_cls
        if min_test_cls < configs.meta_val_query:
            sample_info[3] = min_test_cls

        if dataset == "coswara" and label =="covid":
            sample_info = [configs.meta_val_iter, n_cls[task] * 2, configs.meta_val_shot // 2, configs.meta_val_query]
            # all_data = AudioDataset((np.array(x_data_train + x_data_test), np.array(x_metadata_train + x_metadata_test), np.array(y_label_train + y_label_test)),  from_audio=from_audio, prompt=prompt)
            sampler = TrainCategoriesSampler(group_idxs, *sample_info)
        else:
            sampler = TrainCategoriesSampler(y_label_train, *sample_info)
        
        train_data = AudioDataset((x_data_train, x_metadata_train, y_label_train),  from_audio=from_audio, prompt=prompt)
        test_data = AudioDataset((x_data_test, x_metadata_test, y_label_test),  from_audio=from_audio, prompt=prompt)
        val_data = AudioDataset((x_data_vad, x_metadata_vad, y_label_vad),  from_audio=from_audio, prompt=prompt)
        
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
        train_data = AudioDataset((x_data_train, x_metadata_train, y_label_train),  from_audio=from_audio, prompt=prompt)
        test_data = AudioDataset((x_data_test, x_metadata_test, y_label_test),  from_audio=from_audio, prompt=prompt)
        val_data = AudioDataset((x_data_vad, x_metadata_vad, y_label_vad),  from_audio=from_audio, prompt=prompt)
        train_loader = DataLoader(
            train_data, batch_size=configs.batch_size, num_workers=2, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=configs.batch_size, num_workers=2, shuffle=True
        )
        test_loader = DataLoader(
            test_data, batch_size=configs.batch_size, shuffle=False, num_workers=2
        )
        return train_loader, val_loader, test_loader


def get_prompt(configs, dataset="ssbpr", label="snoring", modality="snoring"):
    data_description = {
        "ssbpr": "This data comes from a snore-based sleep body position recognition dataset (SSBPR).",
        "covid19sounds": "This data comes from the COVID-19 Sounds dataset.",
        "coviduk": "This data comes from the UK COVID-19 Vocal Audio Dataset.",
        "icbhidisease": "This data comes from the ICBHI Respiratory Sound Database Dataset.",
        "coswara": "This data comes from the Coswara Covid-19 dataset. ",
        "kauh": "This data comes from the KAUH lung sound dataset, containing lung sounds recorded from the chest wall using an electronic stethoscope.",
        "copd": "This data comes from the RespiratoryDatabase@TR dataset, containing auscultation sounds recorded from the chest wall using an electronic stethoscope.",
        "coughvid": "This data comes from the CoughVID dataset. ",
    }
    task_description = {
        "snoring": "body position of the participant",
        "symptom": "whether the participant has respiratory symptoms (dry cough, wet cough, fever, sore throat, shortness of breath, runny nose, headache, dizziness, or chest tightness)",
        "covid": "whether the participant has COVID-19",
        "copd": " whether the person has Chronic obstructive pulmonary disease (COPD)",
        "smoker": "whether the person is a smoker or not",
        "obstructive": "whether the person has obstructive respiratory disease including asthma and COPD",
        "copdlevel": "the severity of the COPD patient",
        "asthma": " whether the person has asthma",

    }
    classes = {
        "snoring": "supine, supine but left lateral head, supine but right lateral head, left-side lying, right-side lying",
        "symptom": "symptomatic, asymptomatic",
        "covid": "COVID19, non-COVID19",
        "copd": "COPD, healthy",
        "smoker": "smoker, non-smoker",
        "obstructive": "obstructive, healthy",
        "copdlevel": "COPD 0, COPD 1, COPD 2, COPD 3, COPD 4",
        "asthma": "asthma, healthy"

    }
    n_cls = len(classes[label].split(","))
    if configs.use_audio:
        prompt = (
                    f"<|start_prompt|>"
                    f"Dataset description: {data_description[dataset]} "
                    f"Task description: classifiy {task_description[label]} given the following information and audio of the person's {modality} sounds. "
                    f"The {n_cls} classes are: {classes[label]}. "
                    f"Please output the class index, from 0 to {n_cls-1}."
                    "<|<end_prompt>|>"
                )
    else:
        prompt = (
            f"<|start_prompt|>Dataset description: {data_description[dataset]} "
            f"Task description: classifiy {task_description[label]} given the following information."
            f"The {n_cls} classes are: {classes[label]}. <|<end_prompt>|>"
        )
    return prompt


def get_context(metadata):
    context = ""
    if "gender" in metadata:
        context += "Gender: {}. ".format(metadata["gender"])
    if "age" in metadata:
        context += "Age: {}. ".format(metadata["age"])
    # if "location" in metadata:
    #     context += "Recording location: {}. ".format(metadata["location"])
    if "device" in metadata:
        context += "Recording device: {}. ".format(metadata["device"])
    
    if "vaccination" in metadata:
        context += "Vaccination status: {}. ".format(metadata["vaccination"])

    if "location" in metadata:
        l = metadata["location"]
        if len(l) == 2:
            location_dict = {
                "1": "posterior-upper lung", 
                "2": "posterior-middle lung", 
                "3": "posterior-lower lung", 
                "4": "posterior-inferior lung",
                "5": "posterior-costophrenic angle lung",
                "6": "anterior-lower lung",
                # ICBHI
                'Tc': 'trachea', 
                'Al': 'left anterior chest',
                'Ar': 'right anterior chest', 
                'Pl': 'left posterior chest', 
                'Pr': 'right posterior chest', 
                'Ll': 'left lateral chest', 
                'Lr': 'right lateral chest',
            }
            if l[1].isnumeric():
                location = "left" if l[0] == "L" else "right"
                location += location_dict[l[1]]
            else:
                location = location_dict[l]
        else:
            l = l.replace(" ", "")
            l = l.replace("PLR", "PRL")
            l1_dict = {"P" : "posterior", "A": "anterior"}
            l2_dict = {"L": "left", "R": "right"}
            l3_dict = {"U": "upper", "M": "middle", "L": "lower"}
            try:
                location = l1_dict[l[0]]
                location += " " + l2_dict[l[1]]
                location += " " + l3_dict[l[2]]
            except:
                print(l)

        context += f"Record location: {location}."

    if "medhistory" in metadata:
        userMedHistory = metadata["medhistory"]
        # print(userMedHistory)
        med_history_dict = { 
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
            'valvular': "Valvular heart disease",
            "respiratory_condition_asthma": "asthma",
            "respiratory_condition_other": "other respiratory health condition"
            }
        if pd.isna(userMedHistory) or userMedHistory == "" or "None" in userMedHistory or "none" in userMedHistory:
            # print(row)
            context += "Patient presents with no medical history conditions. "
        elif "pnts" in userMedHistory:
            pass
        else:
            if userMedHistory[0] == ",": userMedHistory = userMedHistory[1:]
            if userMedHistory[-1] == ",": userMedHistory = userMedHistory[:-1]
            context += "Patient presents with the following medical history conditions: " 
            # print(userMedHistory)
            context += ", ".join([med_history_dict[med].lower() for med in userMedHistory.split(",")])  + ". "
    
    if "symptoms" in metadata:
        userSymptoms = metadata["symptoms"]
        symptoms_dict = { 
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
            "loss_of_taste": "loss of sense of taste",
            # coswara
            "cold": "cold", 
            "cough": "cough", 
            "fever":"fever", 
            "diarrhoea": "diarrhoea",
            "st": "sore throat", 
            "loss_of_smell": "loss of smell and/or taste", 
            "mp": "muscle pain", 
            "ftg": "fatigue", 
            "bd": "breathing difficulties",
            # coughvid
            "fever_muscle_pain": "fever or muscle pain"
            }

        if pd.isna(userSymptoms) or userSymptoms == "" or "None" in userSymptoms:
            # print(row)
            context += "Patient presents with no obvious respiratory symptoms."
        elif "pnts" in userSymptoms:
            pass
        else:
            if userSymptoms[0] == ",": userSymptoms = userSymptoms[1:]
            context += "Patient presents with the following respiratory symptoms: " 
            context += ", ".join([symptoms_dict[sym].lower() for sym in userSymptoms.split(",")])  + ". "

    # print(context)
    return context


def downsample_balanced_dataset(x_train, metadata_train, y_train):
    # Find unique classes in y_train
    classes = np.unique(y_train)

    # Find the minimum number of samples among classes
    min_samples = min(np.bincount(y_train))

    # Initialize lists to store downsampled data
    x_downsampled = []
    metadata_downsampled = []
    y_downsampled = []

    # Downsample each class
    for c in classes:
        # Get indices of samples belonging to class c
        indices = np.where(y_train == c)[0]

        # Randomly select min_samples samples
        selected_indices = np.random.choice(
            indices, min_samples, replace=False)

        # Add selected samples to downsampled data
        x_downsampled.extend(x_train[selected_indices])
        metadata_downsampled.extend(metadata_train[selected_indices])
        y_downsampled.extend(y_train[selected_indices])

    # Convert lists to numpy arrays
    x_downsampled = np.array(x_downsampled)
    metadata_downsampled = np.array(metadata_downsampled)
    y_downsampled = np.array(y_downsampled)

    return x_downsampled, metadata_downsampled, y_downsampled


def upsample_balanced_dataset(x_train, metadata_train, y_train):
    # print(x_train.shape, metadata_train.shape, y_train.shape)
    from sklearn.utils import resample, shuffle

    # Separate the dataset into classes
    class_0 = x_train[y_train == 0]
    metadata_0 = metadata_train[y_train == 0]
    class_1 = x_train[y_train == 1]
    metadata_1 = metadata_train[y_train == 1]

    # Find the size of the larger class
    size_0 = len(class_0)
    size_1 = len(class_1)
    max_size = max(size_0, size_1)

    # Upsample the smaller class
    if size_0 < size_1:
        # print(metadata_0.shape)
        class_0_upsampled, metadata_0_upsampled = resample(class_0, metadata_0, replace=True, n_samples=max_size, random_state=42)
        # print(metadata_0_upsampled.shape)
        class_1_upsampled, metadata_1_upsampled = class_1, metadata_1
        # print(metadata_1_upsampled.shape)
        y_class_0 = np.zeros(max_size)
        y_class_1 = y_train[y_train == 1]
    else:
        class_1_upsampled, metadata_1_upsampled = resample(class_1, metadata_1, replace=True, n_samples=max_size, random_state=42)
        class_0_upsampled, metadata_0_upsampled = class_0, metadata_0
        y_class_1 = np.ones(max_size)
        y_class_0 = y_train[y_train == 0]

    # Combine the upsampled classes
    x_train_upsampled = np.concatenate((class_0_upsampled, class_1_upsampled))
    metadata_upsampled = np.concatenate((metadata_0_upsampled, metadata_1_upsampled))
    y_train_upsampled = np.concatenate((y_class_0, y_class_1))

    # print(metadata_upsampled.shape)

    # Shuffle the upsampled dataset
    x_train_upsampled, metadata_upsampled, y_train_upsampled = shuffle(x_train_upsampled, metadata_upsampled, y_train_upsampled, random_state=42)

    print("Balanced dataset sizes:")
    print(f"Class 0: {len(y_train_upsampled[y_train_upsampled == 0])}")
    print(f"Class 1: {len(y_train_upsampled[y_train_upsampled == 1])}")
    return x_train_upsampled, metadata_upsampled, y_train_upsampled



def test(model, test_loader, loss_func, n_cls, plot_feature="", plot_only=False, return_auc=False, verbose=True):
    total_loss = []
    test_step_outputs = []
    features = []
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    with torch.no_grad():
        for i,  (x1, x2, x3, y) in enumerate(test_loader):
            x1 = x1.to(device)
            # print(x3)
            y = y.to(device)
            # print(n_cls, y)
            y_hat = model(x1, x2, x3)
            if plot_feature: 
                feature = model(x1, x2, x3, no_fc=True)
                features.append(feature.detach().cpu().numpy())
            if plot_only:
                test_step_outputs.append((y.detach().cpu().numpy(), None, None))
                continue
            loss = loss_func(y_hat, y)
            total_loss.append(loss.item())

            _, predicted = torch.max(y_hat, 1)
            probabilities = F.softmax(y_hat, dim=1)
            test_step_outputs.append((y.detach().cpu().numpy(), predicted.detach().cpu().numpy(), probabilities.detach().cpu().numpy() ))
    
    all_outputs = test_step_outputs
    y = np.concatenate([output[0] for output in all_outputs])
    if plot_feature:
        features = np.concatenate(features, axis=0)
        plot_tsne(features, y, title=plot_feature)

    if plot_only:
        return
    
    total_loss = np.average(total_loss)
    

    predicted = np.concatenate([output[1] for output in all_outputs])
    probs = np.concatenate([output[2] for output in all_outputs])

    # print(y)
    # print(probs[11])

    acc = np.mean(predicted == y)

    auroc = AUROC(task="multiclass", num_classes=n_cls)
    auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))

    if verbose:
        print("loss", total_loss)
        print("acc", acc)
        print("auc", auc)

    if return_auc:
        return acc, auc

    return total_loss / (i+1)


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def warp_tqdm(data_loader):
    # if args.disable_tqdm:
    #     tqdm_loader = data_loader
    # else:
    tqdm_loader = tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def load_checkpoint(model, configs, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(configs.save_path))
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(configs.save_path))
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    model.load_state_dict(checkpoint['state_dict'])
