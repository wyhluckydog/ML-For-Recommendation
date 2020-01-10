(1)实验的主要任务：使用DeepFM模型的推荐算法完成在数据集上的预测
(2)对应参考的论文为：
“DeepFM:A Factorization-Machine based Neural Network for CTR Prediction”
(3)部署环境：
python3.7
torch1.3
(4)代码结构：
对数据集进行预处理，按照字段合并为一个数据文件dataset
文件夹utils中的split_train.py用于对dataset进行按比例划分，生成训练集和测试集
文件夹utils中的data_preprocess.py对数据集进行预处理，提取连续特征和分类特征，生成category_emb.csv
DeepFM.py是包含了DeepFM的模型代码
main.py为主程序，调用DeepFM模型进行训练
(5)参数：
训练epoch=300,学习率learning-rate=0.003,embedding-size=32,optimizer选用Adam
(6)数据集：Movielens Dataset(CTR预测用的是Criteo Dataset)
dataset下载链接：https://grouplens.org/datasets/movielens/
训练集和测试集比例为8：2
(7)评价指标：AUC和Logloss
实验结果：AUC-0.798605,Logloss-0.50583

