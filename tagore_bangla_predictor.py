import os
import re
from collections import defaultdict, Counter
import nltk
from nltk.corpus import indian

# Download NLTK's built-in Indian language corpus
nltk.download('indian', quiet=True)

# -------------------------------------------------------------------
# 1. Load Built-in NLTK Bangla Corpus & Save Clean Text File
# -------------------------------------------------------------------
def load_bangla_corpus(filepath="tagore_bangla.txt"):
    """
    Loads NLTK's built-in Bangla corpus from 'nltk.corpus.indian',
    strips XML tags and POS tags (_NN, _VM, etc.), and saves clean text locally.
    """
    try:
        raw_text = indian.raw('bangla.pos')
        # 1. Strip XML tags like <Sentence id=...>
        clean_text = re.sub(r'<[^>]+>', '', raw_text)
        # 2. Strip POS tags like _NN, _VM, _PRP
        clean_text = re.sub(r'_[A-Z0-9_\-\.]+', '', clean_text)
        # 3. Clean extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Save clean text locally
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(clean_text)
            print(f"Clean Bangla corpus saved locally to '{filepath}'.")

        return clean_text
    except Exception as e:
        print(f"Error loading NLTK Indian corpus: {e}")
        print("Falling back to built-in sample text...")
        return """
        চিত্ত যেথা ভয়শূন্য, উচ্চ যেথা শির, জ্ঞান যেথা মুক্ত, যেথা গৃহের প্রাচীর।
        """

# -------------------------------------------------------------------
# 2. Bangla Preprocessing & Tokenization Pipeline
# -------------------------------------------------------------------
def preprocess_bangla_corpus(text):
    """
    Extracts Bangla words and isolates punctuation marks (including '।')
    using Unicode regex range \u0980-\u09FF.
    """
    text = re.sub(r'\s+', ' ', text)
    tokens = re.findall(r'[\u0980-\u09FF]+|[।.,!?]', text)
    return tokens

# -------------------------------------------------------------------
# 3. N-Gram Language Model Compiler
# -------------------------------------------------------------------
def build_ngram_model(tokens, n=3):
    """
    Builds an N-gram conditional probability lookup table:
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
# 4. Bangla Next-Word Prediction & Ranking
# -------------------------------------------------------------------
def predict_next_words(history_tokens, model, n=3, top_k=5):
    """
    Retrieves top_k next-word predictions given preceding history.
    Includes fallback to single last word if exact history is unseen.
    """
    if not history_tokens:
        return []

    # 1. Exact (n-1) history match
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
# 5. Multi-Sentence Generation & Sequence Probability Ranking
# -------------------------------------------------------------------
def generate_sentence_from_branch(initial_tokens, initial_prob, model, n=3, max_length=15):
    """
    Generates a sentence along a specific candidate branch and calculates cumulative sequence probability.
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

        if best_next_word in {'।', '৷', '॥', '.', '!', '?'}:
            break

    sentence_str = " ".join(tokens)
    sentence_str = re.sub(r'\s+([।.,!?])', r'\1', sentence_str)
    return sentence_str, cum_prob

def predict_multiple_sentences(seed_text, model, n=3, max_length=15, num_suggestions=3):
    """
    Predicts multiple candidate sentences starting from seed_text and sorts them by probability.
    """
    seed_tokens = preprocess_bangla_corpus(seed_text)
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
# 6. Interactive Terminal Input -> Output File Interface
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("==========================================================")
    print("   Built-in NLTK Bangla Sentence Prediction System        ")
    print("==========================================================")

    raw_text = load_bangla_corpus()
    tokens = preprocess_bangla_corpus(raw_text)
    print(f"Corpus loaded successfully. Total tokens: {len(tokens)}")

    N_GRAM_SIZE = 3
    print(f"Building {N_GRAM_SIZE}-gram Language Model...")
    model = build_ngram_model(tokens, n=N_GRAM_SIZE)
    print("Model compilation complete.\n")

    output_filename = "output.txt"

    # Initialize the output file with main header
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("      Bangla Sentence Prediction Results                  \n")
        f.write("==========================================================\n\n")

    print(f"Ready! All predictions will be written to '{output_filename}'.")

    while True:
        user_input = input("\nEnter starting word/phrase in Bangla (or 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            print("Exiting program.")
            break
        if not user_input:
            continue

        input_tokens = preprocess_bangla_corpus(user_input)

        # Append formatted output directly to output.txt
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"INPUT WORD / PHRASE: {user_input}\n")
            f.write("-" * 50 + "\n")

            f.write("--- Next Word Suggestions (Probabilities) ---\n")
            next_words = predict_next_words(input_tokens, model, n=N_GRAM_SIZE, top_k=5)
            if next_words:
                for word, prob in next_words:
                    f.write(f"  P({word} | {' '.join(input_tokens)}) = {prob:.4f}\n")
            else:
                f.write("  শব্দটি করপাসে পাওয়া যায়নি (No predictions found).\n")

            f.write("\n--- Predicted Sentences (Sorted by Probability) ---\n")
            ranked_sentences = predict_multiple_sentences(
                user_input, model, n=N_GRAM_SIZE, max_length=15, num_suggestions=3
            )

            for idx, (sentence, prob) in enumerate(ranked_sentences, start=1):
                f.write(f"  [{idx}] {sentence}\n")
                f.write(f"      Sequence Probability: {prob:.6e}\n")

            f.write("\n" + "=" * 50 + "\n\n")

        print(f"-> Output added to '{output_filename}'. Open VS Code tab to view.")