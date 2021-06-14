import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


seed = 42
real_news_path = 'D:\\matti\\PycharmProjects\\fake_news\\True.csv'
fake_news_path = 'D:\\matti\\PycharmProjects\\fake_news\\Fake.csv'

# Read the data
fake = pd.read_csv(fake_news_path)
good = pd.read_csv(real_news_path)

# Get shape and head
print('FAKE NEWS')
print(fake.shape)
print(fake.head())
print('REAL NEWS')
print(good.shape)
print(good.head())

# Add label
label_fake = pd.DataFrame(index=range(fake.shape[0]), columns=['label']).fillna('FAKE')
fake = pd.concat([fake, label_fake], axis=1)
label_real = pd.DataFrame(index=range(good.shape[0]), columns=['label']).fillna('REAL')
good = pd.concat([good, label_real], axis=1)
# Final dataset
df = pd.concat([fake, good], axis=0).sample(frac=1, random_state=seed)
labels = df.label

# Split train-test
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=seed)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
# Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct
# classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Unlike most
# other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing very little
# change in the norm of the weight vector.
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Build confusion matrix
confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

