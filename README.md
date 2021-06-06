# Categorical-word-embedding-with-GCN
GCN-based word embedding model, which utilizes information about the structure of sentences using dependencies parsing.

If we look at a resulting representation of words in vector space, we see that words in one level of the hierarchy are closer than in functional type word embedding models, like Word2Vec. Our categorical word embedding utilizes syntactic as semantic information. The model can complete other word embedding models because of their complementarity. Moreover, our model is a good choice for the representation of sentences since word vectors encode information about the position of a word in the sentences. 

For running the model, you should have text corpus saved in a TXT file and parameters collected in a JSON file. Sea examples of JSON file (config.json).
