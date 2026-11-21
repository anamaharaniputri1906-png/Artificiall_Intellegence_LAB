def classify_message(tweet):
    tweet = tweet.lower()
    if "cancelled" in tweet or "delayed" in tweet:
        return "Service Complaint"
    elif "what time" in tweet or "departure" in tweet:
        return "Information Request"
    else:
        return "Other"


def needs_priority(category):
    """Rule 2: every service complaint is prioritised for support."""
    return category == "Service Complaint"


tweet_a = "My flight has been cancelled"
tweet_b = "What time does flight AB123 depart?"

print(classify_message(tweet_a))
print(classify_message(tweet_b))

# Extra: apply the full rule chain (Rule 1 -> Rule 2) to the original tweets
full_a = "My flight has been cancelled and nobody has contacted me."
for name, text in [("A", full_a), ("B", tweet_b)]:
    cat = classify_message(text)
    print(f"Tweet {name}: {cat} | prioritise = {needs_priority(cat)}")
