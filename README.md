# Categorical-word-embedding-with-GCN
_GCN (graph convolutional network)-based word embedding model, which utilizes information about the structure of sentences using dependencies parsing._

If we look at a resulting representation of words in vector space, we see that words in one level of the hierarchy are closer than in functional type word embedding models, like Word2Vec. Our categorical word embedding utilizes syntactic as semantic information. The model can complete other word embedding models because of their complementarity. Moreover, our model is a good choice for the representation of sentences since word vectors encode information about the position of a word in the sentences. 

For running the model, you should have text corpus saved in a TXT file and parameters collected in a JSON file. Sea examples of JSON file (config.json).

### Example
```python
 with open(corpus_path, 'r') as f:
    corpus = f.readlines()

 with open(file) as f:
    config = json.load(f)
        
 device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

 w2v_gcn = W2VGCN(input_list_name = corpus[:100], emb_dimension=config['emb_dimension'], sent_max_len=config['sent_max_len'],
                        iteration=config['iterations'], initial_lr = config['initial_lr'],
                            dropout = config['dropout'], min_count = config['min_count'], batch_size = config['batch_size'])
 w2v_gcn.train()

 with open(model_name, 'wb') as f:
    torch.save(w2v_gcn, f)
```

where
* corpus_path - path to the corpus of text;
* file - path to the json file with cofiguration;
* model_name - path to directory where is needed to save a model.
