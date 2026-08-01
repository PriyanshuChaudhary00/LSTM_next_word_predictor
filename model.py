import torch.nn as nn


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=150):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embeddings = self.embedding(x)
        _, (final_hidden_state, _) = self.lstm(embeddings)
        return self.fc(final_hidden_state[-1])










