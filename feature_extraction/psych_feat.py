from empath import Empath
import pandas as pd

lexicon = Empath()

def calculate_empath_features(content):
    empath_features = lexicon.analyze(content, categories=["hate", 'family', 'crime', 'optimism', 'violence', 'love', 'sadness', 'emotional', 'joy','negative_emotion', 'positive_emotion'], normalize=True)
    return pd.Series(empath_features)


shuffled_data = pd.read_csv('shuffled_data_new_with_features.csv')
empath_features_df = shuffled_data['content'].apply(calculate_empath_features)
shuffled_data = pd.concat([shuffled_data, empath_features_df], axis=1)
shuffled_data.to_csv('data_with_all_features.csv', index=False)