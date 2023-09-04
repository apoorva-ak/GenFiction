from nltk.corpus import cmudict
d = cmudict.dict()
import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
import pandas as pd

def nsyl(word):
  word = word.text
  return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]] 

def syllable_count(word):
    word = word.text
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


def stanza_features(doc):
    data = nlp(doc)
    noun_count = 0
    adj_count = 0
    verb_count = 0
    adverb_count = 0
    pronoun_count = 0
    conjunctions = 0
    first_person = 0
    second_person = 0
    third_person = 0
    present = 0
    past = 0
    future = 0
    active = 0
    passive = 0
    no_of_sentences = 0
    no_of_words = 0
    no_of_complex_words = 0
    no_of_syl = 0
    no_of_punc = 0

    for sent in data.sentences:
        no_of_sentences += 1
        for wrd in sent.words:
            if wrd.upos == 'PUNCT':
                no_of_punc += 1
            else:
                no_of_words += 1
                try:
                    res = nsyl(wrd)
                    no_of_syl += res[0]
                    if(res[0] >= 3):
                        no_of_complex_words += 1
                except:
                    res = syllable_count(wrd)
                    no_of_syl += res
                    if(res >= 3):
                        no_of_complex_words += 1
                
                if wrd.upos == 'NOUN' or wrd.upos == 'PROPN':
                    noun_count += 1
                if wrd.upos == "ADJ":
                    adj_count += 1
                if wrd.upos == "VERB":
                    verb_count += 1
                if wrd.upos == "ADV":
                    adverb_count += 1
                if wrd.upos == "CCONJ":
                    conjunctions += 1
                if wrd.upos == "PRON":
                    pronoun_count += 1
                if wrd.feats:
                    if "Person=1" in wrd.feats:
                        first_person += 1
                    if "Person=2" in wrd.feats:
                        second_person += 1
                    if "Person=3" in wrd.feats:
                        third_person += 1
                    if "Tense=Past" in wrd.feats:
                        past += 1
                    if "Tense=Pres" in wrd.feats:
                        present += 1
                    if "Tense=Fut" in wrd.feats:
                        future += 1
                    if "Voice=Act" in wrd.feats:
                        active += 1
                    if "Voice=Pass" in wrd.feats:
                        passive += 1     

    #stylistic features
    noun_count = noun_count/no_of_words
    adj_count = adj_count/no_of_words
    verb_count = verb_count/no_of_words
    adverb_count = adverb_count/no_of_words
    conjunctions = conjunctions/no_of_words
    pronoun_count = pronoun_count/no_of_words
    first_person = first_person/no_of_words
    second_person = second_person/no_of_words
    third_person = third_person/no_of_words
    past = past/no_of_words
    present = present/no_of_words
    future = future/no_of_words
    active = active/no_of_words
    passive = passive/no_of_words

    #complexity features
    avg_sentence_length = no_of_words/no_of_sentences
    avg_complex_words = no_of_complex_words/no_of_words
    avg_syl_per_word = no_of_syl/ no_of_words
    flesch_reading_index = 206.835 - (1.015*avg_sentence_length) - (84.6* avg_syl_per_word)
    gunning_fog_index = 0.4*(avg_sentence_length + 100*avg_complex_words)

    return noun_count, adj_count, verb_count, adverb_count, conjunctions, pronoun_count, first_person, second_person, third_person, present, past, future, active, passive, avg_sentence_length, avg_complex_words, avg_syl_per_word, gunning_fog_index, flesch_reading_index


def calculate_features(content):
    # Replace this with your actual implementation of the stanza_features function
    noun_count, adj_count, verb_count, adverb_count, conjunctions, pronoun_count, first_person, second_person, third_person, present, past, future, active, passive, avg_sentence_length, avg_complex_words, avg_syl_per_word, gunning_fog_index, flesch_reading_index = stanza_features(content)
    
    return pd.Series({
        'noun_count': noun_count,
        'adj_count': adj_count,
        'verb_count': verb_count,
        'adverb_count': adverb_count,
        'conjunctions': conjunctions,
        'pronoun_count': pronoun_count,
        'first_person': first_person,
        'second_person': second_person,
        'third_person': third_person,
        'present': present,
        'past': past,
        'future': future,
        'active': active,
        'passive': passive,
        'avg_sentence_length': avg_sentence_length,
        'avg_complex_words': avg_complex_words,
        'avg_syl_per_word': avg_syl_per_word,
        'gunning_fog_index': gunning_fog_index,
        'flesch_reading_index': flesch_reading_index
    })


shuffled_data = pd.read_csv('shuffled_data_new.csv')
new_features = shuffled_data['content'].apply(calculate_features)
shuffled_data = pd.concat([shuffled_data, new_features], axis=1)
shuffled_data.to_csv('shuffled_data_new_with_features.csv', index=False)
