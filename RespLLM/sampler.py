import numpy as np
import torch
from torch.utils.data import Sampler
import collections
import random

class CategoriesSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n_shot, n_query):

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        label = np.array(label)
        self.m_ind = []
        unique = np.unique(label)
        unique = np.sort(unique)
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        print(f"sampler info: n_iter: {self.n_iter}, n_way:{self.n_way}, n_shot:{self.n_shot}, n_query:{self.n_query}.")

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            for c in classes:
                l = self.m_ind[c.item()]
                pos = torch.randperm(l.size()[0])
                # print(l.size())
                batch_gallery.append(l[pos[:self.n_shot]])
                batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]])
            # print(len(batch_gallery[0]))
            # print(len(batch_query[0]))
            batch = torch.cat(batch_gallery + batch_query)
            # print(batch)
            yield batch


class TrainCategoriesSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n_shot, n_query, seed=None):

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot

        self.seed = seed

        label = np.array(label)
        self.m_ind = []
        unique = np.unique(label)
        unique = np.sort(unique)
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        print(f"Training sampler info: n_iter: {self.n_iter}, n_way:{self.n_way}, n_shot:{self.n_shot}.")

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        
        for i in range(self.n_iter):
            batch_gallery = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            for c in classes:
                l = self.m_ind[c.item()]
                pos = torch.randperm(l.size()[0])
                # print(l.size())
                batch_gallery.append(l[pos[:self.n_shot]])
            # print(len(batch_gallery[0]))
            # print(len(batch_query[0]))
            batch = torch.cat(batch_gallery)
            # print(batch)
            yield batch


# class SplitCategoriesSampler(Sampler):

#     def __init__(self, label, split, n_iter, n_way, n_shot, n_query, batch_size=15, debug=False, seed=None):

#         self.n_iter = n_iter
#         self.n_way = n_way
#         self.n_shot = n_shot
#         self.n_query = n_query
#         self.batch_size = batch_size # for all test set

        

#         self.class_to_indices = collections.defaultdict(list)
#         self.classes = set()
#         # print(len(label))
#         # print(len(split))
#         for idx, lbl in enumerate(label):  # Assuming the dataset returns (data, label, split)
#             self.class_to_indices[lbl].append((idx, split[idx]))
#             self.classes.add(lbl)
#         # label = np.array(label)
#         # self.m_ind = []
#         # unique = np.unique(label)
#         # unique = np.sort(unique)
#         # for i in unique:
#         #     ind = np.argwhere(label == i).reshape(-1)
#         #     ind = torch.from_numpy(ind)
#         #     self.m_ind.append(ind)
#         print(f"sampler info: n_iter: {self.n_iter}, n_way:{self.n_way}, n_shot:{self.n_shot}, n_query:{self.n_query}.")
#         if n_query < 0:
#             print("n_query overriden, using entire test set")

#         self.seed = seed
#         # self.n_fold = self
#         self.debug = debug
#         if debug:
#             self.starting_sampling_point = [0] * len(self.class_to_indices)

#     def __len__(self):
#         return self.n_iter

#     def __iter__(self):
#         # print(self.n_query)
#         if self.n_query < 0:
#             test_indices = []
#             for cls, indices in self.class_to_indices.items():
#                 test_indices.extend([idx for idx, split in indices if split == 'test'])
#             random.shuffle(test_indices)
#             # print(len(test_indices))
#             batch_gallery = []
#             for label in self.class_to_indices:
#                 indices = self.class_to_indices[label]
#                 train_indices = [idx for idx, split in indices if split == 'train']
#                 # print("group", label, len(train_indices), self.n_shot)
#                 # sampled_train_indices = random.sample(train_indices, self.n_shot)
#                 if self.debug:
#                     # print("sampling", self.starting_sampling_point[label], self.starting_sampling_point[label] + self.n_shot)
#                     # sampled_train_indices = train_indices[self.starting_sampling_point[label]:self.starting_sampling_point[label] + self.n_shot]
#                     # self.starting_sampling_point[label] += self.n_shot
#                     # if self.starting_sampling_point[label] + self.n_shot > len(train_indices):
#                     #     self.starting_sampling_point[label] = 0
#                     start, end = self.seed * self.n_shot, self.seed * self.n_shot + self.n_shot
#                     if end > len(train_indices):
#                         start, end = 0, self.n_shot
#                     print("sampling", start, end)
#                     sampled_train_indices = train_indices[start: end]
#                 else:
#                     random.seed(self.seed)
#                     sampled_train_indices = random.sample(train_indices, self.n_shot)
#                 batch_gallery.extend(sampled_train_indices)
#             for i in range(0, len(test_indices), self.batch_size):
#                 batch_query = test_indices[i: i+ self.batch_size]
#                 batch = torch.tensor(batch_gallery + batch_query)
#                 # print(batch)
#                 yield batch
#         else:
                    
#             for i in range(self.n_iter):
#                 batch_gallery = []
#                 batch_query = []
#                 # print(self.classes, self.n_way)
#                 selected_classes = random.sample(self.classes, self.n_way)
#                 for label in selected_classes:
#                     indices = self.class_to_indices[label]
#                     train_indices = [idx for idx, split in indices if split == 'train']
#                     test_indices = [idx for idx, split in indices if split == 'test']

#                     # Randomly select the required number of samples from each class
#                     sampled_train_indices = random.sample(train_indices, self.n_shot)

#                     sampled_test_indices = random.sample(test_indices, self.n_query)

#                     batch_gallery.extend(sampled_train_indices)
#                     batch_query.extend(sampled_test_indices)
#                 batch = torch.tensor(batch_gallery + batch_query)

#                 yield batch

class SplitCategoriesSampler(Sampler):

    def __init__(self, label, split, n_iter, n_way, n_shot, n_query, batch_size=16, debug=False, seed=None):

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.batch_size = batch_size # for all test set

        self.class_to_indices = collections.defaultdict(list)
        self.classes = set()
        # print(len(label))
        # print(len(split))
        self.test_indices = []
        for idx, lbl in enumerate(label):  # Assuming the dataset returns (data, label, split)
            if split[idx] == "test":
                self.test_indices.append(idx)
            self.class_to_indices[lbl].append((idx, split[idx]))
            self.classes.add(lbl)
        # label = np.array(label)
        # self.m_ind = []
        # unique = np.unique(label)
        # unique = np.sort(unique)
        # for i in unique:
        #     ind = np.argwhere(label == i).reshape(-1)
        #     ind = torch.from_numpy(ind)
        #     self.m_ind.append(ind)
        print(f"sampler info: n_iter: {self.n_iter}, n_way:{self.n_way}, n_shot:{self.n_shot}, n_query:{self.n_query}.")
        if n_query < 0:
            print("n_query overriden, using entire test set")

        self.seed = seed
        # self.n_fold = self
        self.debug = debug
        if debug:
            self.starting_sampling_point = [0] * len(self.class_to_indices)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        # print(self.n_query)
        if self.n_query < 0:
            # test_indices = []
            # for cls, indices in self.class_to_indices.items():
            #     test_indices.extend([idx for idx, split in indices if split == 'test'])
            # random.shuffle(test_indices)

            test_indices = self.test_indices
            # print(len(test_indices))
            batch_gallery = []
            for label in self.class_to_indices:
                indices = self.class_to_indices[label]
                train_indices = [idx for idx, split in indices if split == 'train']
                # print("group", label, len(train_indices), self.n_shot)
                # sampled_train_indices = random.sample(train_indices, self.n_shot)
                if self.debug:
                    # print("sampling", self.starting_sampling_point[label], self.starting_sampling_point[label] + self.n_shot)
                    # sampled_train_indices = train_indices[self.starting_sampling_point[label]:self.starting_sampling_point[label] + self.n_shot]
                    # self.starting_sampling_point[label] += self.n_shot
                    # if self.starting_sampling_point[label] + self.n_shot > len(train_indices):
                    #     self.starting_sampling_point[label] = 0
                    start, end = self.seed * self.n_shot, self.seed * self.n_shot + self.n_shot
                    if end > len(train_indices):
                        start, end = 0, self.n_shot
                    print("sampling", start, end)
                    sampled_train_indices = train_indices[start: end]
                else:
                    random.seed(self.seed)
                    sampled_train_indices = random.sample(train_indices, self.n_shot)
                batch_gallery.extend(sampled_train_indices)
            for i in range(0, len(test_indices), self.batch_size):
                batch_query = test_indices[i: i+ self.batch_size]
                batch = torch.tensor(batch_gallery + batch_query)
                # print(batch)
                yield batch
        else:
                    
            for i in range(self.n_iter):
                batch_gallery = []
                batch_query = []
                # print(self.classes, self.n_way)
                selected_classes = random.sample(self.classes, self.n_way)
                for label in selected_classes:
                    indices = self.class_to_indices[label]
                    train_indices = [idx for idx, split in indices if split == 'train']
                    test_indices = [idx for idx, split in indices if split == 'test']

                    # Randomly select the required number of samples from each class
                    sampled_train_indices = random.sample(train_indices, self.n_shot)

                    sampled_test_indices = random.sample(test_indices, self.n_query)

                    batch_gallery.extend(sampled_train_indices)
                    batch_query.extend(sampled_test_indices)
                batch = torch.tensor(batch_gallery + batch_query)

                yield batch

