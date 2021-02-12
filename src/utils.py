import pickle

def load_pickle(path:str):
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data

def save_pickle(data, path:str):
    with open(path, "wb") as file:
        pickle.dump(data, file)

def load_tsv(path):
    with open(path,"r") as file:
        lines = [l.strip() for l in file.readlines()]
    lines = [l.split("\t") for l in lines if l != ""]
    return lines

def load_lines(path):
    with open(path, "r") as file:
        lines = [l.strip() for l in file.readlines() if l.strip()!=""]
    return lines