import matplotlib.pyplot as plt
import pandas as pd

# 绘制loss与k的关系图
# cols=['k','loss']
# loss_for_k=pd.read_csv('data/test_loss.csv', encoding='utf-8',names=cols)
# k=loss_for_k["k"]
# loss=loss_for_k['loss']
# epoch=loss.shape[0]
# plt.plot(k,loss,marker='o',label='loss data')
# plt.title('loss for k')
# plt.xlabel('k')
# plt.ylabel('loss')
# plt.legend()
# plt.show()

# 绘制loss与学习率的关系图
cols=['lr','loss']
loss_for_k=pd.read_csv('data/test_loss_for_lr.csv', encoding='utf-8',names=cols)
lr=loss_for_k["lr"]
loss=loss_for_k['loss']
epoch=loss.shape[0]
plt.plot(lr,loss,marker='o',label='loss data')
plt.title('loss for lr')
plt.xlabel('lr')
plt.ylabel('loss')
plt.legend()
plt.show()