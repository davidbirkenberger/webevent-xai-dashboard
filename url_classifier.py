import torch
from torch.nn.functional import sigmoid
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_url_classifier(checkpoint_path: str, device: str = "cpu"):
    '''Load classifier and tokenizer from checkpoint'''
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-german-europeana-cased")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(device)
    model.eval()
    return model, tokenizer


def predict_about_score(texts, model, tokenizer, device="cpu", batch_size=16):
    '''Predict sigmoid scores for a list of URL texts'''
    model.eval()
    model.to(device)

    all_scores = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            logits = model(**inputs).logits.squeeze()
            scores = sigmoid(logits)  # Binary classification → sigmoid
            score_list = scores.cpu().numpy().tolist()
            if not isinstance(score_list, list):
                score_list = [score_list]
            all_scores.extend(score_list)

    return all_scores


def rank_urls_by_about_score(urls, model, tokenizer, device="cpu", top_k=None):
    '''Return a sorted DataFrame of URLs and their aboutness scores'''
    import pandas as pd
    scores = predict_about_score(urls, model, tokenizer, device=device)
    df = pd.DataFrame({"url": urls, "about_score": scores})
    df_sorted = df.sort_values("about_score", ascending=False)
    return df_sorted.head(top_k) if top_k else df_sorted