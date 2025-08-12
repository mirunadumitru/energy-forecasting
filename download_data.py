import requests

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
r = requests.get(url)

with open("data/raw/energydata_complete.csv", "wb") as f:
    f.write(r.content)