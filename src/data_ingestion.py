import pandas as pd
import numpy as np
import os

df = pd.read_csv("https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv")

X = df.drop("category", axis = 1)
y = df["category"]


X["clean_comment"] = X["clean_comment"].apply(lambda x: "".join(str(x).split()).lower())
X = X[X["clean_comment"] != ""]
y = y.loc[X.index]  

os.makedirs("data/raw", exist_ok=True)

X.to_csv("data/raw/X.csv")
y.to_csv("data/raw/y.csv")
