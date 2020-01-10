import torch
import lib
import numpy as np
import os

np.random.seed(22)
torch.manual_seed(22)
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(22)
data_floder = "C:/Users/34340/Desktop/rsc15_dataset/"   #dataset on windows
train_data_file = "train_data.txt" #训练集文件名
test_data_file = "test_data.txt" #测试集文件名
load_model = "C:/Users/34340/Desktop/rsc15_dataset/model_100.pt"

def main():
    print("Loading train data from {}".format(os.path.join(data_floder, train_data_file)))
    print("Loading test data from {}".format(os.path.join(data_floder, test_data_file)))
    train_data = lib.Dataset(os.path.join(data_floder, train_data_file))
    test_data = lib.Dataset(os.path.join(data_floder, test_data_file))

    input_size = len(train_data.items)  #total num of the items, to generate the one-hot
    hidden_size = 100   #hidden dimension in GRU
    num_layers = 3  #layers of the network
    output_size = input_size    #output size equals to the input size
    batch_size = 50 #size of the batch
    dropout_hidden = 0.5    #dropout of the hidden layer
    final_act = 'tanh'  #final activate
    loss_type = 'TOP1-max'  #loss function
    optimizer_type = 'Adagrad'  #optimizer type name
    lr = 0.01   #learning rate
    n_epochs = 100    #epoch of the train process
    loss_function = lib.LossFunction(loss_type=loss_type)   #define the loss function

    print("Choose your action: ")
    print("1、Training      2、Testing      q、Quit")
    user_action = input()
    while user_action not in ['1', '2', 'q']:
        print("Choose your action: ")
        print("1、Training      2、Testing      q、Quit")
        user_action = input()

    if user_action == '1':
        print("Training..................")
        model = lib.GRU4REC(input_size, hidden_size, output_size, final_act=final_act,
                            num_layers=num_layers, use_cuda=cuda, batch_size=batch_size,
                            dropout_hidden=dropout_hidden)
        optimizer = lib.Optimizer(model.parameters(), optimizer_type=optimizer_type, lr=lr)
        trainer = lib.Trainer(model, train_data=train_data, eval_data=test_data, optim=optimizer,
                              use_cuda=cuda, loss_func=loss_function, batch_size=batch_size)
        trainer.train(1, n_epochs)
    elif user_action == '2':
        print("Testing...................")
        if os.path.exists(load_model):
            print("Loading pre_trained model from {}".format(load_model))
            checkpoint = torch.load(load_model)
            model = checkpoint["model"]
            model.gru.flatten_parameters()
            evaluation = lib.Evaluation(model, loss_function, use_cuda=cuda, k=20)
            loss, f1, mrr = evaluation.eval(test_data, batch_size)
            print("Final result：f1 = {:.2f}, mrr = {:.2f}".format(f1, mrr))
        else:
            print("No Pretrained Model was found! Train your model first!")

if __name__ == '__main__':
    main()
