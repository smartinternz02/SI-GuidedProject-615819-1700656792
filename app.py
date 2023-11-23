from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Read the original dataset from the CSV file
zomato_df = pd.read_csv('clean_dataset.csv')

# Assuming zomato_df is your original DataFrame
df_percent = pd.DataFrame(zomato_df)
df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)

# Use a random sample of 10% of the data
sampled_data = df_percent.sample(frac=0.1, random_state=42)

# Handle NaN values in 'reviews_list'
sampled_data['reviews_list'] = sampled_data['reviews_list'].fillna('')

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(sampled_data['reviews_list'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

zomato_df.dropna(subset=['name'], inplace=True)
zomato_df['name'].fillna('Unknown', inplace=True)


def recommend(name,cosine_similarities= cosine_similarities):
  # create a list to put top restaurants
  recommend_restaurant=[]

  # find the index of the hotel entered
  idx =indices[indices==name].index[0]

  # find the restaurant with a similar cosine-sin value
  score_series =pd.Series(cosine_similarities[idx]).sort_values(ascending=False)

  top30_indexes=list(score_series.iloc[0:31].index)

  for each in top30_indexes:
    recommend_restaurant.append(list(df_percent.index)[each])

  df_new=pd.DataFrame(columns=['name','cuisines','Mean Rating','cost'])

  for each in recommend_restaurant:
    df_new =df_new.append(pd.DataFrame(df_percent[['cuisines','Mean Rating','cost']][df_percent.index == each].sample()))

  df_new=df_new.drop_duplicates(subset=['cuisines','Mean Rating','cost'],keep=False)
  df_new=df_new.sort_values(by='Mean Rating', ascending=False).head(10)
  print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS:'%(str(len(df_new)),name))
  df_new = df_new[['name', 'cuisines', 'Mean Rating', 'cost']]


  return df_new.to_dict('records')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    restaurant_name = request.form['restaurant_name']
    recommendations = recommend(restaurant_name)
    return render_template('recommendations.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
