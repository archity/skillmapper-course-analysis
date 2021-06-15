import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def load_and_make_table(file="./data/sample_50_reviews.csv"):
    table_df = pd.read_csv(file, sep=";")
    print("Dataset read:")
    print(f"{table_df.shape[0]} rows and {table_df.shape[1]} coloumns")
    return table_df


def print_table(df):
    print(df)


def clean_the_data(table_df):
    df_clean = table_df

    # Remove punctuation marks from every review
    punc_chars = '''!()-[]{};:'"\,<>./?@#$%^&*_~’‘´`~|+'''
    df_clean.loc[:, "review"] = df_clean.loc[:, "review"].str.translate(str.maketrans('', '', punc_chars))

    # Convert all the letters of words to lowercase
    df_clean["review"] = df_clean["review"].str.lower()

    # print_table(df_clean["review"])
    return df_clean


def top_words_and_wordcloud(df_clean):
    # Remove stopwords
    stopwords_english = stopwords.words('english')
    lang_pat = r'\b(?:{})\b'.format('|'.join(stopwords_english))
    df_clean.loc[:, "review"] = df_clean.loc[:, "review"].str.replace(lang_pat, '')

    # Go through each tweet and put individual word into a list
    word_list = []
    for tweet in df_clean["review"]:
        word_list += (tweet.split())

    # Get the count values of all the words
    words_counter = Counter(word_list).most_common()

    # Convert the Counter list to a Pandas dataframe
    words_counter_df = pd.DataFrame.from_records(list(dict(words_counter).items()), columns=['word', 'count'])

    # Write the list to a txt file
    with open("./data/words_counter_list.txt", 'w') as f:
        f.write(words_counter_df.to_string(index=False))
    # print_table(words_counter_df)

    # Wordcloud
    # Convert pd dataframe to dictionary for input for wordcould
    wordcount_dict = dict(zip(words_counter_df["word"], words_counter_df["count"]))
    my_dpi = 200
    wc = WordCloud(width=1920, height=1080).generate_from_frequencies(wordcount_dict)
    fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    if not os.path.exists("img"):
        os.makedirs("img")
    plt.savefig("img/wordcloud.png", format="png", dpi=my_dpi)


if __name__ == '__main__':
    table_df = load_and_make_table()
    df_clean = clean_the_data(table_df)
    top_words_and_wordcloud(df_clean)
