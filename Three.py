from __future__ import print_function

import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def get_data():
    if os.path.exists("hearthead.csv"):
        print("-- hearthead.csv found locally")
        df = pd.read_csv("hearthead.csv")

    return df

df = get_data()
print("* df.head()", df.head(), sep="\n", end="\n\n")
print("* df.tail()", df.tail(), sep="\n", end="\n\n")

def encode_target(df, target_column):

    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

print("* Has heart desease: 1 - Yes; 0 - No", df["value"].unique(), sep="\n")
df2, targets = encode_target(df, "value")

features = list(df2.columns[:13])
print("* features:", features, sep="\n")

y = df2["Target"]
X = df2[features]
dt = DecisionTreeClassifier(min_samples_split=20)
dt.fit(X, y)

def get_code(tree, feature_names, target_names,
             spacer_base="    "):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)

get_code(dt, features, targets)
