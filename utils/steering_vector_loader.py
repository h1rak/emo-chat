import glob
import pickle
from tqdm import tqdm
import random

def load_activations(vector_path):
    with open(f'{vector_path}/activations_train.pkl','rb') as f:
        train = pickle.load(f)

    with open(f'{vector_path}/activations_test.pkl','rb') as f:
        test = pickle.load(f)

    return train,test