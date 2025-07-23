import pandas as pd
import numpy as np

data = pd.read_csv("clean_weather.csv", index_col = 0)
data = data.ffill()
print(data['tmax'].head(10))


