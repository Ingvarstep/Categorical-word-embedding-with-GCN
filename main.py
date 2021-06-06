from w2v_model import W2VGCN
import torch
import numpy as np
import json

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def main():
    print('Please write the path to the corpus txt file.')
    corpus_path = input()
    with open(corpus_path, 'r') as f:
        corpus = f.readlines()
    print('Please write the path to arguments stored in the JSON file.')
    file = input()
    with open(file) as f:
        config = json.load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Creating of the model and starting processing text data...')
    w2v_gcn = W2VGCN(input_list_name = corpus[:100], emb_dimension=config['emb_dimension'], sent_max_len=config['sent_max_len'],
                        iteration=config['iterations'], initial_lr = config['initial_lr'],
                            dropout = config['dropout'], min_count = config['min_count'], batch_size = config['batch_size'])
    print('Training...')
    w2v_gcn.train()

    print('Saving the model...')
    print('Please write file name for storing the model.')
    model_name = input()
    with open(model_name, 'wb') as f:
        torch.save(w2v_gcn, f)

if __name__ == "__main__":
    main( )
