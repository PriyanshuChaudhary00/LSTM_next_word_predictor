from torch.utils.data import Dataset , DataLoader
import torch
import numpy as np
def get_data(path):
    sentences = []

    with open(path) as txt:
        for i in txt:
            sentences.append(i.strip())
        
    def tokenize(text):
        text = text.lower()
        text = text.replace("?" , "")
        text = text.replace("'" , "")
        return text.split()

    vocab = {"<unk>":0}

    for i in sentences:
        x = tokenize(i)
        for j in x:
            if j not in vocab:
                vocab[j] = len(vocab)
    sentences.remove("")
    sentences.remove('\ufeff')

    def text_to_index(text):
        indexed_text = []
        for j in tokenize(text):
            if j in vocab:
                indexed_text.append(vocab[j])
            else :
                indexed_text.append(vocab["<unk>"])
        return indexed_text

    def sentence_to_data(x):
        sentences = []
        for i in x:
            indexed_text = text_to_index(i)
            sentences.append(indexed_text)
        return sentences

    number_sentence = sentence_to_data(sentences)

    training_sequence = [
        sen[:i+1]
        for sen in number_sentence
        for i in range(1, len(sen))
    ]

    # Pre-compute max length once — avoids O(n²) recomputation inside the loop
    max_len = max(len(seq) for seq in training_sequence)

    # Vectorized padding using numpy — much faster than Python list concat
    padded = np.zeros((len(training_sequence), max_len), dtype=np.int64)
    for i, seq in enumerate(training_sequence):
        padded[i, max_len - len(seq):] = seq
        # if i == 1000 : break
    padded_training_sequence = torch.from_numpy(padded)

    x = padded_training_sequence[: , :-1]
    y = padded_training_sequence[: , -1]

    class Custom_dataset(Dataset):
        def __init__(self , x, y):
            self.x = x
            self.y = y

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self , index):
            return self.x[index] , self.y[index]

    dataset = Custom_dataset(x , y)

    data_loader = DataLoader(dataset , batch_size=64 , shuffle=True)
    
    return data_loader , vocab , text_to_index