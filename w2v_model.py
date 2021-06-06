import torch
import numpy as np
import time
from tqdm import tqdm
import torch.optim as optim

from text_preprocessing import GCN_Data
from GraphConvolution import GCN

class W2VGCN():
    def __init__(self,
                 input_list_name,
                 emb_dimension=50,
                 sent_max_len=50,
                 iteration=20,
                 initial_lr=0.001,
                 dropout = 0.5,
                 min_count = 1,
                 batch_size = 100):
        self.sent_max_len = sent_max_len
        self.data = GCN_Data(input_list_name, min_count, sent_max_len)
        self.adjM = self.data.AdjM
        self.fM = self.data.fM
        self.ohM = self.data.ohM
        # self.ohMn = self.data.ohMn
        self.emb_size = len(self.data.word2id)
        self.sent_count = self.data.sentence_count
        self.emb_dimension = emb_dimension
        self.n_feature = len(self.data.deps_dict)
        self.batch_size = batch_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.dropout = dropout
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.gcn_model = GCN(self.n_feature, self.emb_dimension, self.emb_size, self.sent_max_len, self.dropout).to(self.device)

        self.optimizer = optim.Adam(self.gcn_model.parameters(), lr=self.initial_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.sent_count)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def train(self):
        for i in tqdm(range(self.iteration)):
            order = np.random.permutation(self.sent_count)
            loss = 0
            for start_index in range(0, self.sent_count, self.batch_size):

                self.optimizer.zero_grad()
                self.gcn_model.train()

                sLapM = self.adjM[start_index:start_index+self.batch_size]
                LapM = torch.cat([lapm.to_dense().unsqueeze(0) for lapm in sLapM], dim=0)
                f_batch = self.fM[start_index:start_index+self.batch_size]
                f_batch = torch.cat(f_batch, dim = 0)
                oh_batch = self.ohM[start_index:start_index+self.batch_size]
                oh_batch = torch.cat(oh_batch, dim=0)


                loss_train = 0

                input1 = f_batch.long().to(self.device)
                input2 = LapM.to(self.device)
                target = oh_batch.long().to(self.device)

                output = self.gcn_model.forward(input1, input2)
                output_t = output.transpose(1, 2).contiguous()
                # print(output.shape, target.shape)
                loss_train = self.loss_func(output_t, target)
                loss_train.backward()

                self.optimizer.step()
                self.scheduler.step()
                loss += loss_train
            if i%1 == 0 :
                print('Epoch:', i, 'Loss:', loss)

    def save_embedding(self, output_path):
        model_emb = [i.cpu().data for i in self.gcn_model.parameters()][1]
        model_info = [model_emb, self.data.word2id, self.data.id2word]
        meFile = open(output_path, 'wb')
        pickle.dump(model_info, meFile)
        meFile.close()

    def word_sim(self, word, i, top_n=10):
        w1_index = self.data.word2id[word]
        weights = [i.cpu().data for i in self.gcn_model.parameters()][1].numpy().transpose()
        v_w1 = weights[w1_index]
        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.data.word_count):
            v_w2 = weights[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den
            word = self.data.id2word[i]
            word_sim[word] = theta
        words_sorted = sorted(word_sim.items(), key= lambda item: item[1], reverse=True)
        return words_sorted
