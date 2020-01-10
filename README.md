# ML-For-Recommendation
### 以下实验为北京邮电大学计算机学院815实验室学生共同完成，主要为recommendation方向的经典模型论文的复现。
#### MF模型：
##### （1）实验主要任务：
> 使用MF（Matrix factorization）在movielens数据集上进行电影评分预测任务
##### （2）参考论文：
> 《MATRIX FACTORIZATION TECHNIQUES FOR RECOMMENDER SYSTEMS》
##### （3）部署环境：
> python37 + pytorch1.3
##### （4）数据集：
> movielen的small数据集，使用的rating.csv文件。数据集按照8:2的比例进行划分，随机挑选80%的数据当做训练集，剩余的20%当做测试集。
##### （5）代码结构：
> load_rating_data(file_path) 读取数据集并划分训练集和测试集
> class mf() 对user和item分别构造隐藏层m*p和n*p并做矩阵乘法并与已有评分做误差，最小化误差优化。

##### （6）参数的调节：
> 特征因子数目的选取：100
> 习率的选取：0.01
##### （7）评价标准：
> 采用rmse作为评价指标，使用测试集对模型进行测试。

#### BPR模型：
##### （1）实验的主要任务：
> 使用BPR模型在movielen数据集上进行电影推荐
##### （2）参考论文：
> 《Bayesian Personalized Ranking》
##### （3）部署环境：
> python3.7+pytorch1.3
##### （4）数据集：
> movielens的small数据集，使用的rating.csv文件。数据集按照8:2的比例进行划分，随机挑选80%的数据当做训练集，剩余的20%当做测试集。从数据集中选取的特征包括：userId , movieId , rating
##### （5）代码结构：
> class TripletUniformPair(Dataset)：生成userid、itemid、ratings的三元组
> class BPR(nn.Module)：使用矩阵分解模型，最大化后验估计量，生成每个userid对应的top-k itemid的rank，使用adam最优化。
##### （6）参数调节：
> 训练集和测试集的划分：0.8 0.2
> 学习率：0.001
##### （7）评级指标：
> NDCG：0.6553

#### FM(Factorization Machines)
##### （1）实验的主要任务：
> 使用FM在movielen数据集上进行电影评分预测任务（rendle的工作，经典的特征选择）
##### （2）参考论文：
> Factorization Machines
##### （3）部署环境：
> python37 + pytorch1.3
##### （4）数据集：
> Movielen的small数据集（下载链接https://grouplens.org/datasets/movielens/）
使用的rating.csv文件。数据集按照8:2的比例进行划分，随机挑选80%的数据当做训练集，剩余的20%当做测试集。
> 从数据集中选取的特征包括：userId , movieId , lastmovie , rating
> lastmovie数据的构造过程为：将数据集按照userId进行排序，在对于每一个用户按照时间戳进行排序，找出对应于某个电影的上一个电影的movieId。
##### （5）代码结构：
> 进行数据预处理以及数据划分的代码在divideData.py文件中，划分之后得到rating_train.csv与rating_test.csv两个文件。（data文件夹下的ratings.csv为原始数据集，其中会得到一些中间文件：ratings_sort.csv文件为按照useId以及timestamp对数据集排序后得到的文件；rating_addLastMovie.csv文件为增加用户看的上一部电影的movieId得到文件；ratingsNoHead.csv文件为去掉数据集的表头得到的文件。）
> fm_model.py文件是读取训练集以及测试集，并使用pytorch框架编写FM训练模型，最后使用rmse作为评价指标，使用测试集对模型进行测试。模型训练过程中采用batch对数据集进行分批训练，同时每训练完一轮之后使用测试集进行测试，检验测试效果，并最终以曲线的形式展现出来。最终训练集与测试集的曲线图如下图所示：

##### （6）参数的调节：
> ①特征因子k的选取：在test_loss_for_k.py文件中含有绘制loss与k的关系图的代码，通过观察曲线的走向，选取合适的k值（前提是要先将loss与对应的k的数据存储存储到csv文件中，对应为data文件夹下的test_loss.csv文件）
> ②学习率的选取，同样在test_loss_for_k.py文件中含有绘制loss与学习率的关系图的代码，通过观察曲线的走向，选取合适的lr值。（同样应该将loss与学习率lr的对应曲线存储到csv文件中，对应于data文件夹下的test_loss_for_lr.csv文件）
> 同样的训练次数、正则化次数也是通过这种方法进行选取。
##### （7）评价标准：
> 采用rmse作为评价指标，使用测试集对模型进行测试。（实验只使用了数据集中的一部分数据，同样也使用了完整的数据集进行了测试，测试误差为1.2。由于数据集较大，这里只上传使用的部分数据集。）

#### NCF：
##### 1. 实验主要任务
> 使用深度学习模型NCF编写一个电影评分预测系统（xiangnan He的论文，第一个深度学习用在推荐系统的论文）
##### 2. 参考论文
> Neural Collaborative Filtering(Xiangnan He)
##### 3. 部署环境
> Python 3.7.3
> pytorch 1.1.0
##### 4. 代码结构
> DataPreprocess.py	处理原始数据集，并将数据集分成训练集、验证集和测试集。同时使用随机负采样。
>    Dataset.py	加载保存在文件中的数据集。
>    evaluate.py	评估模型方法。
>    GMF.py	MF模型的实现
>   MLP.py	多层感知机模型的实现
>    NeuMF.py	MF+MLP的结合模型实现
##### 5. 数据集
>    使用tafeng数据集。
>    （1） 将tafeng数据集中的CUSTOMER_ID和PRODUCT_ID映射到0-n（n为用户数）和0-m（m为商品数）之间。
>    （2） 然后将数据集转换成论文使用的格式
>    （3） 将处理后的数据随机分成训练集，验证集和测试集（根据论文中的验证集每个用户只有1条），并使用随机负采样。
##### 6. 评价标准
>   Hit Ratio、NDGC

#### DeepFM：
##### (1)实验的主要任务：
> 使用DeepFM模型的推荐算法完成在数据集上的预测
##### (2)对应参考的论文为：
> “DeepFM:A Factorization-Machine based Neural Network for CTR Prediction”
##### (3)部署环境：
> python3.7
> torch1.3
##### (4)代码结构：
> 对数据集进行预处理，按照字段合并为一个数据文件dataset
> 文件夹utils中的split_train.py用于对dataset进行按比例划分，生成训练集和测试集
> 文件夹utils中的data_preprocess.py对数据集进行预处理，提取连续特征和分类特征，生成category_emb.csv
> DeepFM.py是包含了DeepFM的模型代码
> main.py为主程序，调用DeepFM模型进行训练
##### (5)参数：
> 训练epoch=300,学习率learning-rate=0.003,embedding-size=32,optimizer选用Adam
##### (6)数据集：
> Movielens Dataset（数据集下载链接：https://grouplens.org/datasets/movielens/）	(原论文中用的是Criteo Dataset)
> 训练集和测试集比例为8：2
##### (7)评价指标：AUC和Logloss
> 实验结果：AUC：0.798605,Logloss：0.50583

#### GRURec：
##### 1、实验主要任务：
> 	该实验为GRU4Rec论文复现实验，主要是对于用户消费商品的时序推荐。通过用户之前购买的商品记录，进行下一可能购买商品的推荐。
##### 2、参考论文：
> 	《Session-based Recommendations with Recurrent Neural Networks》ICLR 2016
##### 3、环境部署：
> 	python版本3.7
>	pytorch版本为1.3.0
> 	cudatoolkit=10.1
##### 4、代码结构：
	GRU4REC
		--lib
			--__init__.py
			--dataset.py
			--evaluation.py
			--lossfunction.py
			--metric.py
			--model.py
			--optimizer.py
			--trainer.py
		--main.py
		--preprocessing.py
##### 5、关于超参数的选择问题：
>	本实验的超参数主要是在论文提出的超参数的基础上进行的微调，如若需要修改超参数，在main.py中的main函数中可以进行修改
##### 6、数据集：
>	数据集使用到的是论文中使用的RecSys Challenge 2015的数据集yoochoose-click。(https://2015.recsyschallenge.com/)
>	数据集的处理：
>>>		该数据集提供 会话id（用户id）、用户与商品交互的时间、商品id 信息。
>>>		其中测试集是所有交互时间最晚的一个交互记录的前24小时内的有交互记录的用户的所有交互记录。（也就是说把最后一个交互记录前一天内有交互记录的用户的所有交互记录取出）
>>>		训练集是除去测试集后所有的用户交互记录
>>>		并且需要去掉不足2个交互记录的会话对应的交互记录，去掉不足5个交互记录的商品对应的交互记录
>>>		该处理过程在preprocess.py中完成
##### 7、评价标准：
>	f1、mrr
##### 8、训练过程中：
>	每训练5个epoch，会使用测试集对模型进行测试，并且会画出一张训练集和测试集基于每个epoch的mrr对比图，便于查看模型的训练情况


#### meta-path:
##### (1)实验主要任务：
> 利用元数据(Metapath)完成对用户电影评分预测
##### (2)参考论文：
> Collaborative Filtering with Entity Similarity Regularization in Heterogeneous Information Networks
##### (3)部署环境：
> Python 3.7+pytorch 1.3
##### (4)代码结构：
> movieknowledgehandle.ipynb文件用于对数据集进行处理，得到如下处理文件：
>> ①movie_actor.txt   ②movie_director.txt  ③movie_genres.txt  ④movie_tag.txt   ⑤train.txt  ⑥test.txt
>> utils.py中load_data读入数据，并生成相关knowledge的邻接矩阵，运行main.py即可训练。
##### (5)超参数的选择
> 在main.py中设定参数
##### (6)数据集：
> 数据集MovieLens + IMDb（下载链接https://grouplens.org/datasets/hetrec-2011/，下载的数据集为：hetrec2011-movielens-2k-v2.zip），用户80%的数据作为训练集，20%的数据作为测试集。
##### (7)评价标准：
> 评价标准为rmse

