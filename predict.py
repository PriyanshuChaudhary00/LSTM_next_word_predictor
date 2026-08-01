from model import LSTM_Model
from dataset import get_data
import torch 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

path = "data1.txt"
data_loader , vocab , text_to_index = get_data(path)

model = LSTM_Model(len(vocab))
model.load_state_dict(torch.load("checkpoint/next-word-predictor.pth"))
model.to(device)
def prediction(model , vocab , text):
  num_text = text_to_index(text)

  padded_text = torch.tensor([0] * (18 - len(num_text)) + num_text , dtype=torch.long).unsqueeze(0)
  padded_text = padded_text.to(device)

  model.eval()

  logit = model(padded_text)
  value , index = torch.max(logit ,dim=1)
  final_output = list(vocab.keys())[index]
  return final_output
 
# pred = prediction(model , vocab , "hello how are ")
# print(pred)

token = int(input("Enter number of token: "))
text = input("Enter Your text:  ")
pred = text 

for i in range(token):
    pred += " " + prediction(model , vocab , pred)
print(pred)