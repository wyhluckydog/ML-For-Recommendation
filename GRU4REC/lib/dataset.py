import pandas as pd
import numpy as np
import torch

#读取数据
class Dataset(object):
    def __init__(self, path, sep=',', session_key='SessionID', item_key='ItemID', time_key='Time'):
        self.df = pd.read_csv(path, sep=sep, dtype={session_key: int, item_key: int, time_key: float})
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.add_item_indices()     #根据item的种类个数添加一列item_idx列
        self.df.sort_values([self.session_key, self.time_key], inplace=True)    #根据sessionID排序，再按时间排序
        self.click_offsets = self.get_click_offsets()   #每个session的开始记录的位置
        self.session_idx_arr = self.order_session_idx() #session个数的标号

    def add_item_indices(self):
        item_ids = self.df[self.item_key].unique()
        item2idx = pd.Series(data=np.arange(len(item_ids)),
                             index=item_ids)
        itemmap = pd.DataFrame({self.item_key: item_ids, 'item_idx': item2idx[item_ids].values})
        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')

    def get_click_offsets(self):
        offsets = np.zeros(self.df[self.session_key].nunique()+1, dtype=np.int32)
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        return offsets

    def order_session_idx(self):
        session_idx_arr = np.arange(self.df[self.session_key].nunique())
        return session_idx_arr

    @property
    def items(self):
        return self.itemmap[self.item_key].unique()

#以batch的大小返回每个session当前的item_idx
class DataLoader():
    def __init__(self, dataset, batch_size=50):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        df = self.dataset.df
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        maxiter = iters.max()   #batch当前走到的最大的session
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = []
        finished = False

        while not finished:
            minlen = (end - start).min()    #最短的session长度
            idx_target = df.item_idx.values[start]  #每个session开始位置的item的idx

            for i in range(minlen - 1):
                idx_input = idx_target  #batch大小，模型的输入item的idx
                idx_target = df.item_idx.values[start + i + 1]  #batch大小，目标item的idx（即：标签）
                input = torch.LongTensor(idx_input) #转为torch的tensor
                target = torch.LongTensor(idx_target)
                yield input, target, mask   #返回：输入、标签、需要清空的hidden

            start = start + (minlen-1)
            mask = np.arange(len(iters))[(end - start) <= 1]    #找出所有走完了的session
            for idx in mask:
                maxiter += 1    #batch当前走到的session
                if maxiter >= (len(click_offsets) - 1):   #如果batch走完了所有session，就退出
                    finished = True
                    break
                iters[idx] = maxiter    #更新iters对应位置的session
                start[idx] = click_offsets[session_idx_arr[maxiter]]    #下一个session的开始位置
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]  #下一个session的结束位置