import os
import re
import urllib.request
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK tokenizers
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# -------------------------------------------------------------------
# 1. Corpus Loading (Rabindranath Tagore's Gitanjali)
# -------------------------------------------------------------------
def load_tagore_corpus(filepath="tagore_gitanjali.txt"):
    """
    Downloads Rabindranath Tagore's Gitanjali (Project Gutenberg eBook #3160).
    """
    if not os.path.exists(filepath):
        print("Downloading Rabindranath Tagore's 'Gitanjali' corpus...")
        url = "https://www.gutenberg.org/cache/epub/3160/pg3160.txt"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        )
        try:
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Download failed ({e}). Using built-in sample text.")
            return """
            Thou hast made me endless, such is thy pleasure. This frail vessel thou emptiest 
            again and again, and fillest it ever with fresh life. This little flute of a reed 
            thou hast carried over hills and dales, and hast blown through it melodies eternally new. 
            At the immortal touch of thy hands my little heart loses its limits in joy and gives 
            birth to utterance ineffable. Thy infinite gifts come to me only on these very small 
            hands of mine. Ages pass, and still thou pourest, and still there is room to fill.
            """

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Strip Gutenberg metadata headers/footers
    start_idx = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end_idx = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]

    return text

# -------------------------------------------------------------------
# 2. Text Preprocessing
# -------------------------------------------------------------------
def preprocess_corpus(text):
    """
    Cleans corpus text while keeping punctuation and stop words intact 
    for sequence language modeling.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    tokens = word_tokenize(text)
    return tokens

# -------------------------------------------------------------------
# 3. N-Gram Language Model Compiler
# -------------------------------------------------------------------
def build_ngram_model(tokens, n=3):
    """
    Builds an N-gram conditional probability table:
    prob_model[history_tuple][next_word] = P(next_word | history_tuple)
    """
    counts = defaultdict(Counter)

    for i in range(len(tokens) - n + 1):
        history = tuple(tokens[i : i + n - 1])
        next_word = tokens[i + n - 1]
        counts[history][next_word] += 1

    prob_model = defaultdict(dict)
    for history, context in counts.items():
        total_count = sum(context.values())
        for next_word, count in context.items():
            prob_model[history][next_word] = count / total_count

    return prob_model

# -------------------------------------------------------------------
# 4. Next-Word Prediction & Ranking
# -------------------------------------------------------------------
def predict_next_words(history_tokens, model, n=3, top_k=5):
    """
    Retrieves top_k next-word predictions given the preceding history.
    Includes smart fallback if exact (n-1) context is missing.
    """
    if not history_tokens:
        return []

    # 1. Try exact (n-1) history match
    history_key = tuple(history_tokens[-(n - 1):])
    if history_key in model:
        candidates = model[history_key].items()
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]

    # 2. Fallback: match histories ending with the last word
    last_word = history_tokens[-1]
    fallback_counts = Counter()
    for history, next_words in model.items():
        if history[-1] == last_word:
            for word, prob in next_words.items():
                fallback_counts[word] += prob

    if fallback_counts:
        total = sum(fallback_counts.values())
        normalized = [(word, count / total) for word, count in fallback_counts.items()]
        return sorted(normalized, key=lambda x: x[1], reverse=True)[:top_k]

    return []

# -------------------------------------------------------------------
# 5. Multi-Sentence Generation & Probability Ranking
# -------------------------------------------------------------------
def generate_sentence_from_branch(initial_tokens, initial_prob, model, n=3, max_length=15):
    """
    Generates a sentence along a specific candidate branch and calculates sequence probability.
    """
    tokens = list(initial_tokens)
    cum_prob = initial_prob

    for _ in range(max_length - len(tokens)):
        history_key = tuple(tokens[-(n - 1):])

        if history_key not in model:
            history_key = tuple(tokens[-1:])
            if history_key not in model:
                break

        candidates = sorted(model[history_key].items(), key=lambda x: x[1], reverse=True)
        best_next_word, prob = candidates[0]

        tokens.append(best_next_word)
        cum_prob *= prob

        if best_next_word in {'.', '!', '?'}:
            break

    sentence_str = " ".join(tokens)
    sentence_str = re.sub(r'\s+([.,!?;:])', r'\1', sentence_str)
    return sentence_str.capitalize(), cum_prob

def predict_multiple_sentences(seed_text, model, n=3, max_length=15, num_suggestions=3):
    """
    Generates multiple sentence completions and sorts them by total probability.
    """
    seed_tokens = preprocess_corpus(seed_text)
    if not seed_tokens:
        return []

    top_branches = predict_next_words(seed_tokens, model, n=n, top_k=num_suggestions)

    if not top_branches:
        return []

    results = []
    for next_word, prob in top_branches:
        branch_tokens = seed_tokens + [next_word]
        sentence, sentence_prob = generate_sentence_from_branch(
            branch_tokens, prob, model, n=n, max_length=max_length
        )
        results.append((sentence, sentence_prob))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# -------------------------------------------------------------------
# 6. Interactive Interface
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("==========================================================")
    print("      Tagore Corpus Sentence Prediction System            ")
    print("==========================================================")

    raw_text = load_tagore_corpus()
    tokens = preprocess_corpus(raw_text)
    print(f"Corpus loaded successfully. Total tokens: {len(tokens)}")

    N_GRAM_SIZE = 3
    print(f"Building {N_GRAM_SIZE}-gram Language Model...")
    model = build_ngram_model(tokens, n=N_GRAM_SIZE)
    print("Model compilation complete.\n")

    while True:
        user_input = input("\nEnter a starting word/phrase (or 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            break
        if not user_input:
            continue

        input_tokens = preprocess_corpus(user_input)

        print("\n--- Next Word Suggestions (Probabilities) ---")
        next_words = predict_next_words(input_tokens, model, n=N_GRAM_SIZE, top_k=5)
        if next_words:
            for word, prob in next_words:
                print(f"  P({word} | {' '.join(input_tokens)}) = {prob:.4f}")
        else:
            print("  No predictions found in vocabulary.")

        print("\n--- Predicted Sentences (Sorted by Probability) ---")
        ranked_sentences = predict_multiple_sentences(
            user_input, model, n=N_GRAM_SIZE, max_length=15, num_suggestions=3
        )

        for idx, (sentence, prob) in enumerate(ranked_sentences, start=1):
            print(f"  [{idx}] {sentence}")
            print(f"      Sequence Probability: {prob:.6e}")