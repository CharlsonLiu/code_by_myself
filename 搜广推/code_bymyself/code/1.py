import pandas as pd


path = '搜广推\\fun-rec\\codes\\base_models\\data\\movie_sample.txt'
samples_data = pd.read_csv(path, sep='\t', header = None)
samples_data.columns = ["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id", "label"]

X = samples_data[["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id"]]
y = samples_data["label"]
print(samples_data.head(5))