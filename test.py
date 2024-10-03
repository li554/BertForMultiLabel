from torch.utils import data as Data
from model import Model
import pandas as pd
import pickle
import os

path = "dataset/results.pkl"
model_name = "output/checkpoint-6/"
test_path = "dataset/test_merge.csv"
save_path = 'dataset/test_predict.csv'

if os.path.exists(path):
    file = open(path, "rb")
    results = pickle.load(file)
    file.close()
else:
    model = Model(model_name)
    test_dataset = model.processor.read_dataset(test_path)
    test_sampler = Data.SequentialSampler(test_dataset)
    test_iter = Data.DataLoader(test_dataset, sampler=test_sampler, batch_size=16)
    results = model.predict_iter(test_iter)
    file = open(path, 'wb')
    pickle.dump(results, file)
    file.close()

data = pd.read_csv(test_path)
data.drop('comment_text', axis=1, inplace=True)

for i in range(len(results)):
    for j in range(len(results[0])):
        data.iloc[i, j + 1] = results[i][j].item()

data.to_csv(save_path, index=False)
