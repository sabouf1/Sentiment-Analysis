import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def analyze_sentiment(review):
    """
    Analyze the sentiment of the given review.

 
    """
    #  pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # Preprocess  text
    inputs = tokenizer(review, return_tensors='pt', truncation=True, padding=True)

    # sentiment analysis
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted sentiment class (positive/negative)
    sentiment_class = "positive" if outputs.logits[0, 0] > 0 else "negative"

    sentiment_result = {"sentiment": sentiment_class}
    return sentiment_result

def save_sentiment_as_json(sentiment, filename):
    """
    Save the analysis result as a JSON file.

    """
    with open(filename, "w") as json_file:
        json.dump(sentiment, json_file)

if __name__ == "__main__":
    review_text = input("Enter your review: ")
    sentiment_result = analyze_sentiment(review_text)
    print("Sentiment Analysis Result:")
    print(sentiment_result)

    output_file = "sentiment_result.json"
    save_sentiment_as_json(sentiment_result, output_file)
    print(f"Sentiment analysis result saved as {output_file}")
