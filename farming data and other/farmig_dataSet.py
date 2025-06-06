import pandas as pd
import sys
import csv
import os
import numpy as np

df = pd.read_csv("farming data and other\data\summary_scores.csv")

fulldata = []

for ind in range(df.shape[0]):
    session = df.iloc[ind]
    session_df = pd.read_csv(session["Session File"])
    game_llist = []
    for i in range(session_df.shape[0]):
        listdd = session_df.iloc[i].tolist()
        game_llist.append(listdd)
    fulldata.append([np.array(game_llist), session["Score"]])

print(fulldata[2])