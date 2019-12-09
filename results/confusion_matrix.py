import tensorflow as tf
import json

def load_json():
    path = "/Volumes/Watermelon/results_pc/20191204-134841.json"
    #20191204 - 134841.json
    with open(path) as json_file:
        data = json.load(json_file)
    return data

def compute_confusion():
    pass

if __name__ == '__main__':
    data = load_json()
    print("hola")
