import pandas as pd
import matplotlib.pyplot as plt

# Load movies metadata
movies_df = pd.read_csv("..\\..\\data\\raw\\movies_metadata.csv")

missing_values = movies_df.isnull().sum().sort_values(ascending=False)
print(missing_values.head(10))

# Plot the distribution of movie ratings
plt.hist(movies_df['vote_average'], bins=20)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of movie release years
movies_df['release_year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').dt.year
movies_df['release_year'].hist(bins=50, figsize=(12, 6))
plt.title('Distribution of Movie Release Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.show()

# Plot the top 10 most common movie genres
genres = []
for genre_list in movies_df['genres'].dropna().apply(lambda x: eval(x)):
    genres.extend(genre_list)
genres_df = pd.DataFrame({'genre': genres})
genres_df['genre'].value_counts().head(10).plot(kind='barh', figsize=(12,6))
plt.title('Top 10 Most Common Movie Genres')
plt.xlabel('Frequency')
plt.ylabel('Genre')
plt.show()

# Plot word cloud of movie titles
text = ' '.join(movies_df['title'].dropna().values)
wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Movie Titles')
plt.show()