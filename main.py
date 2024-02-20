
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

fake_news_samples = [
    "Scientists confirm that aliens have landed on Earth!",
    "Breaking: New study claims that eating chocolate can make you lose weight!",
    "Government officials caught in massive corruption scandal!",
    "Famous celebrity spotted with a unicorn in their backyard!",
    "Breaking: Giant asteroid on collision course with Earth!",
]

real_news_samples = [
    "Study finds evidence of water on Mars.",
    "Stock market experiences record gains in the past quarter.",
    "President signs new bill into law to improve healthcare access.",
    "Researchers discover new species in the Amazon rainforest.",
    "Local community comes together to clean up the park.",
]

def train_classifier(fake_samples, real_samples):
    """
    Train a simple classifier using TF-IDF features.
    """
    corpus = fake_samples + real_samples
    labels = ['fake'] * len(fake_samples) + ['real'] * len(real_samples)

    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(corpus)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)

    # Train logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Evaluate classifier accuracy
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Classifier Accuracy:", accuracy)

    return vectorizer, classifier

def get_article_text(url):
    """
    Retrieve the text content of a news article from the given URL.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from <p> tags or any other relevant tags based on the website structure
        article_text = ' '.join([p.get_text() for p in soup.find_all('div', class_='_1W5s')])
        return article_text
    except Exception as e:
        print("Error retrieving article text:", str(e))
        return None

def classify_news_article(article_text, vectorizer, classifier):
    """
    Classify a news article as fake or real.
    """
    # Vectorize the article text
    X_article = vectorizer.transform([article_text])

    # Predict the class (fake or real)
    prediction = classifier.predict(X_article)

    return prediction[0]

def main():
    # Sample dataset for demonstration
    fake_samples = fake_news_samples
    real_samples = real_news_samples

    vectorizer, classifier = train_classifier(fake_samples, real_samples)

    news_url = "https://www.timesnownews.com/latest-news"
    article_text = get_article_text(news_url)

    if article_text:
        print("\nArticle Text:\n", article_text)
        classification = classify_news_article(article_text, vectorizer, classifier)
        print("\nClassification: ", classification)
    else:
        print("Failed to retrieve article text. Please check the URL and try again.")

if __name__ == "__main__":
    main()

