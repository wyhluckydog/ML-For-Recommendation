# coding: utf-8
"""
该文件主要是对数据进行预处理，将评分数据按照8：2分为训练数据与测试数据
"""
import pandas as pd
import csv
import os

#将文件中的数据按照userId进行排序，如果userId相同则按照timestamp进行排序
origin_f = open('data/ratings.csv','rt',encoding='utf-8',errors="ignore")
new_f= open('data/ratings_sort.csv','wt',encoding='utf-8',errors="ignore",newline="")
reader=csv.reader(origin_f)
writer=csv.writer(new_f)
sortedlist=sorted(reader,key=lambda x:(x[0],x[3]),reverse=True)
print(sortedlist.__len__())
i=0
for row in sortedlist:
    if i==0:
        # 添加表头
        writer.writerow(('userId','movieId','rating','timestamp'))
    if(i==(sortedlist.__len__()-1)):
        continue
    writer.writerow(row)
    i=i+1
origin_f.close()
new_f.close()

#增加last_movie字段
csvfile1=open('data/ratings_sort.csv','rt')
reader=csv.DictReader(csvfile1)
rows=[row for row in reader]
# print(rows[0].get("userId"))
i=0
csvfile2=open('data/ratings_sort.csv','rt')
reader=csv.DictReader(csvfile2)
rating_addLastMovie=open('data/rating_addLastMovie.csv','wt',encoding='utf-8',errors="ignore",newline="")
writer=csv.writer(rating_addLastMovie)
for row in reader:
    # 添加用户看得上一部电影
    if i<(rows.__len__()-1) and rows[i].get("userId")==rows[i+1].get("userId"):
        row["last_movie"]=rows[i+1].get("movieId")
    else:
        row["last_movie"]=0
    # print(row.values())
    if i==0:
        writer.writerow(('userId', 'movieId', 'rating', 'timestamp','last_movie'))
    writer.writerow(row.values())
    i=i+1
csvfile1.close()
csvfile2.close()
rating_addLastMovie.close()

# 删除文件中的表头
origin_f = open('data/rating_addLastMovie.csv','rt',encoding='utf-8',errors="ignore")
new_f = open('data/ratingsNoHead.csv','wt+',encoding='utf-8',errors="ignore",newline="")
reader = csv.reader(origin_f)
writer = csv.writer(new_f)
i=0
for i,row in enumerate(reader):
    if i>1:
        writer.writerow(row)
origin_f.close()
new_f.close()

#将数据按照8:2的比例进行划分得到训练数据集与测试数据集
df = pd.read_csv('data/ratingsNoHead.csv', encoding='utf-8')
# df.drop_duplicates(keep='first', inplace=True)  # 去重，只保留第一次出现的样本
# print(df)
df = df.sample(frac=1.0)  # 全部打乱
cut_idx = int(round(0.2 * df.shape[0]))
df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
# 打印数据集中的数据记录数
print(df.shape,df_test.shape,df_train.shape)
# print(df_train)
# 将数据记录存储到csv文件中
# 存储训练数据集
df_train=pd.DataFrame(df_train)
df_train.to_csv('data/ratings_train_tmp.csv',index=False)
# 由于一些不知道为什么的原因，使用pandas读取得到的数据多了一行，在存储时也会将这一行存储起来，所以应该删除这一行（如果有时间在查一查看能不能解决这个问题）
origin_f = open('data/ratings_train_tmp.csv','rt',encoding='utf-8',errors="ignore")
new_f = open('data/ratings_train.csv','wt+',encoding='utf-8',errors="ignore",newline="")     #必须加上newline=""否则会多出空白行
reader = csv.reader(origin_f)
writer = csv.writer(new_f)
for i,row in enumerate(reader):
    if i>0:
        writer.writerow(row)
origin_f.close()
new_f.close()
os.remove('data/ratings_train_tmp.csv')
# 存储测试数据集
df_test=pd.DataFrame(df_test)
df_test.to_csv('data/ratings_test_tmp.csv',index=False)
origin_f = open('data/ratings_test_tmp.csv','rt',encoding='utf-8',errors="ignore")
new_f = open('data/ratings_test.csv','wt+',encoding='utf-8',errors="ignore",newline="")
reader = csv.reader(origin_f)
writer = csv.writer(new_f)
for i,row in enumerate(reader):
    if i>0:
        writer.writerow(row)
origin_f.close()
new_f.close()
os.remove('data/ratings_test_tmp.csv')
