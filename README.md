# 📚 Amazon Book Review Sentiment Classifier

This project performs binary sentiment classification (positive or negative) on Amazon book reviews using both traditional NLP methods and deep learning techniques. The best-performing model is deployed as an interactive web app using Streamlit.

## 📌 Project Overview

- **Dataset**: Amazon book reviews (review summaries only)
- **Goal**: Classify reviews as Positive or Negative
- **Techniques**: VADER, RoBERTa, Dense Neural Network, CNN, LSTM
- **Deployment**: Streamlit Cloud

## 🧹 Data Preprocessing

- Removed null, empty, and duplicate entries
- Retained key columns: `Id`, `Title`, `review/score`, `review/summary`
- Normalized text (lowercasing, punctuation removal)
- Labeled sentiment using VADER (`compound >= 0.05 → Positive`, `<= -0.05 → Negative`)

## 🔍 Baseline Models

| Model     | Type          | Notes                            |
|-----------|---------------|----------------------------------|
| VADER     | Rule-based    | Fast and interpretable           |
| RoBERTa   | Transformer   | Context-aware, more accurate     |

## 🧠 Neural Network Models

Three models were implemented:

- **Dense + Flatten**: Basic feedforward network
- **CNN**: Best performer, captures local word patterns
- **LSTM**: Captures sequential dependencies

### 📊 Performance Summary

| Model           | Train Acc | Test Acc | Test Loss |
|----------------|-----------|----------|-----------|
| Dense + Flatten| 90.46%    | 90.08%   | 0.253     |
| CNN            | 97.25%    | 94.01%   | 0.161     |
| LSTM           | 88.06%    | 88.05%   | 0.354     |

## 🚀 Deployment with Streamlit

The CNN model is deployed using Streamlit and hosted on Streamlit Cloud.

### 🔧 App Features

- Input a book review summary
- Returns sentiment prediction with confidence score
- Clean and minimal UI

```python
# Sample prediction logic
sequence = tokenizer.texts_to_sequences([user_input])
padded = pad_sequences(sequence, maxlen=100, padding='post')
prediction = model.predict(padded)[0][0]
label = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
```

## 🛠️ Tech Stack

- Python, TensorFlow, Keras
- NLTK (VADER), Hugging Face Transformers
- Streamlit (for app deployment)
- Pandas, NumPy, Matplotlib (for data handling & EDA)

## 📈 Future Work

- Extend to multi-class sentiment (e.g. neutral, mixed)
- Fine-tune RoBERTa on domain-specific reviews
- Incorporate full `review/text` for deeper context
- Add feedback loop and mobile-friendly UI

## 📚 References

- [VADER: Hutto & Gilbert, 2014](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)
- [RoBERTa: Liu et al., 2019](https://arxiv.org/abs/1907.11692)
- [Streamlit](https://streamlit.io)
- [TensorFlow](https://www.tensorflow.org)
- [Amazon Product Review Dataset](https://nijianmo.github.io/amazon/index.html)
