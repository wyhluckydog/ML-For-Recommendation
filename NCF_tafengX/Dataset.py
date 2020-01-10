from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

class DatasetLoder(object):
    def __init__(self, path, ):

        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape
        print('DatasetLoader __init__ finished!')
        '''
        self.traindict = self.load_rating_file_as_list(path + ".train.rating")
        self.testdict = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testdict["data"]) == len(self.testNegatives)
        self.num_users = self.traindict["size"][0]
        self.num_items = self.traindict["size"][1]
        self.trainDataset = CustomDataset(self.traindict["data"])
        self.testRatings = CustomDataset(self.testdict["data"])
        self.testNegatives = CustomDataset(self.testNegatives)
        '''

    def get_train_loader(self, batchsize, num_negatives):
        #users = []
        #items = []
        #labels = []
        trainlist = []
        for (u, i) in self.trainMatrix.keys():
            #users.append(u)
            #items.append(i)
            #labels.append(float(1))
            trainlist.append((u, i, float(1)))
            for t in range(num_negatives):
                j = np.random.randint(self.num_items)
                while (u, j) in self.trainMatrix.keys():
                    j = np.random.randint(self.num_items)
                #users.append(u)
                #items.append(j)
                #labels.append(float(0))
                trainlist.append((u, j, float(0)))
        traindataset = CustomDataset(trainlist)
        print('DatasetLoader get_train_loader finished!')
        return DataLoader(traindataset,batchsize,True)

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        print('Datasetloader load_rating_file_as_matrix finished!')
        return mat



    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        print('Datasetloader load_negative_file finished!')
        return negativeList

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        print('Datasetloader load_rating_file_as_list finished!')
        return ratingList

    '''
    def load_rating_file_as_list(self, filename):
  
        # Get number of users and items
        num_users, num_items = 0, 0
        traindict = {}
        trainList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i, r = int(arr[0]), int(arr[1]), float(arr[2])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                if r > 0:
                    trainList.append((u, i, float(1)))
                line = f.readline()
        sizelist = []
        sizelist.append(num_users + 1)
        sizelist.append(num_items + 1)
        traindict["size"] = sizelist
        traindict["data"] = trainList
        return traindict
    '''

if __name__ == '__main__':
    dt = DatasetLoder("Data/ml-1m")
    print(dt.num_items)
    print(dt.num_users)
    print(dt.trainDataset.dataset)
