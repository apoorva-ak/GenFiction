import pandas as pd
from sklearn.utils import shuffle

real_data = pd.read_csv('real_fiction_data.csv')
generated_data = pd.read_csv('generated_fiction_data.csv')
real_data = real_data.iloc(200)[['content']]
real_data['isReal'] = 1
generated_data['isReal'] = 0
merged_data = pd.concat([real_data, generated_data])
shuffled_data = shuffle(merged_data)
shuffled_data.reset_index(drop=True, inplace=True)
shuffled_data.to_csv('shuffled_data_new.csv', index=False)