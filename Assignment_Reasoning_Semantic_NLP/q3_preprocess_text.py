import re


def preprocess_text(text):
    # Replace usernames with USER
    text = re.sub(r"@\w+", "USER", text)
    # Replace URLs with URL
    text = re.sub(r"https?://\S+", "URL", text)
    # Convert text to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


tweet = "@SkyAir My flight is NOT good! It was delayed again. https://airline.example"
clean_tweet = preprocess_text(tweet)
print(clean_tweet)
