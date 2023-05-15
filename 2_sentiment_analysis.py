import pandas as pd
import pysentiment2 as ps
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import time

# Set parameters
# tfidf threshold
threshold = 0.07
# threshold for word cloud
threshold_wc = 0.01

# Set score save path and file name
filename = f"C:/Users/kaiyi/OneDrive/Desktop/Kernow Asset/KERNOWAM/score_threshold_{threshold}.csv"

# Set word cloud save path and file name
save_path = "C:/Users/kaiyi/OneDrive/Desktop/Kernow Asset/KERNOWAM/wordcloud"

# read main files
df = pd.read_csv('final_df_sampled.csv')
ind50 = pd.read_csv('ind50.csv')

# Create a new DataFrame
sc = pd.DataFrame(columns=['RIC', 'date', 'Polarity', 'Positive', 'Negative', 'Neutral', 'Subjectivity'])

# Instantiate a sentiment analyzer
lm = ps.LM()

# Define TF-IDF vectorized
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# calculate all TF-IDF and lm score then save
for key, value in df.iterrows():

    # get RIC, date and text_body
    RIC = value['RIC']
    date = value['date']
    text = value['text_body']

    if pd.isnull(text):
        text = 'This is some text with a NaN value'
    else:
        text = str(text)

    # cal TF-IDF value
    tfidf = tfidf_vectorizer.fit_transform([text])
    idf = tfidf_vectorizer.idf_

    # Get feature names and corresponding TF-IDF values
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_values = tfidf.toarray()[0]

    # Select important features based on TF-IDF value
    important_features = [feature_names[i] for i in range(len(feature_names)) if tfidf_values[i] > threshold]
    print(important_features)

    # Perform sentiment analysis on important features
    tokens = lm.tokenize(" ".join(important_features))
    print(tokens)
    score = lm.get_score(tokens)

    # get sentiment score
    polarity_score = score.get('Polarity', 0)
    positive_score = score.get('Positive', 0)
    negative_score = score.get('Negative', 0)
    subjectivity_score = score.get('Subjectivity', 0)

    # Calculate neutral score
    neutral_score = 1 - positive_score - negative_score

    # Show score
    print(f"  Polarity：{score['Polarity']}")
    print(f"  Positive：{score['Positive']}")
    print(f"  Negative：{score['Negative']}")
    print(f"  Neutral：{neutral_score}")
    print(f"  Subjectivity：{score['Subjectivity']}")

    # Store the scores into a DataFrame
    sc = sc.append(
        {'RIC': RIC, 'date': date, 'Polarity': polarity_score, 'Positive': positive_score, 'Negative': negative_score,
         'Neutral': neutral_score, 'Subjectivity': subjectivity_score}, ignore_index=True)

    # Save scores to csv file
    sc.to_csv(filename, index=False)

    # Check for file write completion to prevent access disallowed conditions in loops
    while True:
        try:
            with open(filename, 'r') as file:
                file_contents = file.read()
            break
        except PermissionError:
            # The file is still occupied by other processes or programs, wait for a while and try again
            time.sleep(0.1)


# Make word cloud plot for 50 top company
# Set RIC attribute in df as index
df = df.set_index('RIC')

# Merge ind50 and df data frames, merge according to RIC attributes
merged_df = pd.merge(ind50, df, on='RIC')

# Merge the text_body of the same RIC into a string and generate a word cloud
for ric, group in df.groupby('RIC'):
    text1 = ' '.join(group['text_body'].values)

    # Calculate TF-IDF value
    tfidf = tfidf_vectorizer.fit_transform([text1])
    idf = tfidf_vectorizer.idf_

    # Get feature names and corresponding TF-IDF values
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_values = tfidf.toarray()[0]

    # Select important features based on TF-IDF value
    important_features = [feature_names[i] for i in range(len(feature_names)) if tfidf_values[i] > threshold_wc]
    important_features_str = ' '.join(important_features)

    # Perform sentiment analysis on important features
    tokens = lm.tokenize(" ".join(important_features))
    tokens_str = ' '.join(tokens)

    wc_tfidf = WordCloud(background_color="white", max_words=2000, contour_width=3, contour_color='steelblue', collocations=True, scale=5).generate(important_features_str)
    # Save word cloud
    wc_tfidf.to_file(f"{save_path}{ric}_tfidf.png")

    wc_tfidf_lm = WordCloud(background_color="white", max_words=2000, contour_width=3, contour_color='steelblue', collocations=True, scale=5).generate(
        tokens_str)
    # Save word cloud
    wc_tfidf_lm.to_file(f"{save_path}{ric}_seg.png")

