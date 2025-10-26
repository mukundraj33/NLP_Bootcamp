# NLP Bootcamp - Assignment 1: Advanced Text Preprocessing

## 📋 Overview
This assignment implements a comprehensive **advanced text preprocessing pipeline** for Natural Language Processing using the Amazon Fine Food Reviews dataset. The implementation demonstrates production-level preprocessing techniques suitable for real-world NLP applications.

## 🚀 Features Implemented

### 🔧 Basic Preprocessing
- **Lowercasing** - Convert all text to lowercase
- **Punctuation Removal** - Clean special characters and symbols
- **Stopword Removal** - Remove common words with extended stopword list
- **URL Removal** - Extract and remove web links
- **HTML Tag Removal** - Clean HTML markup from text
- **Stemming** - Reduce words to their root form using Porter Stemmer
- **Lemmatization** - Convert words to base forms using WordNet Lemmatizer
- **Tokenization** - Split text into tokens with robust error handling

### 📊 Advanced NLP Techniques
- **Bag of Words (BoW)** - Word frequency representation with n-grams
- **TF-IDF** - Term frequency-inverse document frequency
- **One-Hot Encoding** - Categorical data encoding
- **Word2Vec** - Word embeddings using Skip-gram model
- **Performance Benchmarking** - Time analysis of preprocessing steps

## 📊 Dataset

- **Source**: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) from Kaggle
- **Size**: ~500,000 product reviews
- **Features**: Review text, ratings, product information, user metadata
- **Complexity**: Real-world, noisy text data with varied writing styles


# NLP Bootcamp - Assignment 2: Neural Network Sentiment Classification

## 📋 Overview
This assignment implements a **Feedforward Neural Network** for text sentiment classification using PyTorch. The model classifies Yelp reviews into three sentiment categories (Negative, Neutral, Positive) using Bag-of-Words features and advanced neural network architecture with dropout regularization.

## 🎯 Assignment Objectives
- Implement a neural network for text classification
- Preprocess text data using tokenization and stemming
- Convert text to numerical features using Bag-of-Words
- Train and evaluate a PyTorch neural network model
- Compare model performance with and without dropout

## 🧠 Model Architecture

### Neural Network Structure
```python
FeedforwardNeuralNetModel(
  (fc1): Linear(in_features=VOCAB_SIZE, out_features=500)
  (relu1): ReLU()
  (fc2): Linear(in_features=500, out_features=500)
  (relu2): ReLU()
  (fc3): Linear(in_features=500, out_features=3)
)
FeedforwardNeuralNetModelWithDropout(
  (fc1): Linear(in_features=VOCAB_SIZE, out_features=500)
  (relu1): ReLU()
  (dropout1): Dropout(p=0.3)
  (fc2): Linear(in_features=500, out_features=500)
  (relu2): ReLU()
  (dropout2): Dropout(p=0.3)
  (fc3): Linear(in_features=500, out_features=3)
)
```
# 📊 Dataset

**Source**: Yelp Reviews Subset  
**File**: `yelp_reviews_subset_2.csv`  
**Classes**: 3 sentiment categories  
**Training-Test Split**: 70% Training, 30% Testing  

### Sentiment Mapping
| Stars | Sentiment | Encoded Label |
|-------|-----------|---------------|
| 1-2 | Negative | -1 → 0 |
| 3 | Neutral | 0 → 1 |
| 4-5 | Positive | 1 → 2 |

