import numpy as np
import torch

def load_data():
    print('loading data...')
    f1 = open('E:/MetaPath/datatxt/movie_actor.txt')
    f2 = open('E:/MetaPath/datatxt/movie_director.txt')
    f3 = open('E:/MetaPath/datatxt/movie_genres.txt')
    f4 = open('E:/MetaPath/datatxt/movie_tag.txt')
    f5 = open('E:/MetaPath/datatxt/train.txt')
    f6 = open('E:/MetaPath/datatxt/test.txt')
    movie_dict = {}
    movie_list = []
    director_dict = {}
    director_list = []
    actor_dict = {}
    genres_dict = {}
    tag_dict = {}
    user_dict = {}
    user_list = []
    movie_director = {}
    use_movie_score_list = []

    #print(movie_director_adj)
    #print(movie_dict)
    #print(director_dict)
    #print(len(movie_director))

    #print(len(director_list))
    #print(str[0])
    #print(str[1])
    for line in f5:
        str = line.split()
        use_movie_score_list.append(str)
        if str[0] not in user_list:
            user_list.append(str[0])
        if str[1] not in movie_list:
            movie_list.append(str[1])

    for i in range(len(user_list)):
        user_dict.update({user_list[i]:i})

    for i in range(len(movie_list)):
        movie_dict.update({movie_list[i]:i})

    #print('movie_list_len:', len(movie_list))
    #print(len(movie_list))

    Y_user_movie_adj = np.zeros((len(user_list), len(movie_list)))
    R_user_movie_score = np.zeros((len(user_list), len(movie_list)))
    for item in use_movie_score_list:
        Y_user_movie_adj[user_dict[item[0]]][movie_dict[item[1]]] = 1
        R_user_movie_score[user_dict[item[0]]][movie_dict[item[1]]] = float(item[2])


    for line in f2:
        str = line.split()
        if str[0] in movie_list:
            movie_director.update({str[0]: str[1]})
            if str[1] not in director_list:
                director_list.append(str[1])
    #做director_dict
    for i in range(len(director_list)):
        director_dict.update({director_list[i]: i})
    # 做movie_director_adj
    movie_director_adj = np.zeros((len(movie_list), len(director_list)))
    for item in movie_director:
        movie_director_adj[movie_dict[item]][director_dict[movie_director[item]]] += 1

    #print(Y_user_movie_adj)
    #print(R_user_movie_score)
    #print(movie_director_adj)
    test_user_list =[]
    test_movie_list = []
    label = []
    for line in f6:
        str = line.split()
        if str[1] in movie_list:
            test_user_list.append(str[0])
            test_movie_list.append(str[1])
            label.append(str[2])

    user_index = []
    movie_index = []
    for item in test_user_list:
        user_index.append(user_dict[item])
    for item in test_movie_list:
        movie_index.append(movie_dict[item])

    return len(user_list),len(movie_list), movie_list, Y_user_movie_adj, R_user_movie_score,movie_director_adj,user_index, movie_index,label



def pathsim(A : np.ndarray):
    S = np.dot(A, A.transpose())
    return S

def laplacian(S : np.ndarray):
    len = S.shape[0]
    diag_list = []
    for i in range(len):
        diag_list.append(np.sum(S[i]))
    L = np.diag(np.array(diag_list))-S
    return L

def evaluation(U, V, user_index, movie_index, label ):
    #U_arr = U.numpy()
    #V_arr = V.numpy()
    sum = 0
    #user_index_arr = np.array(user_index )
    #movie_index_array
    #label_arr = np.array(label)
    pred = []
    for i in range(len(user_index)):
        pred_label = float(torch.sum(U[user_index[i]]*V[movie_index[i]]))
        pred.append(pred_label)
    #pred_arr = np.array(pred )
    for j in range(len(label)):
        sum += (pred[j]- float(label[j]))**2
    rmse = (sum/len(label))**0.5
    return rmse







