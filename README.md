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


# NLP Bootcamp - Assignment 3: Sentiment Classification using RNNs & LSTMs

## 📋 Overview
This assignment implements **advanced sentiment classification** using Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Bidirectional LSTM models. The implementation includes a comprehensive comparison of different neural architectures and ensemble methods for text sentiment analysis.

## 🎯 Assignment Objectives
- Implement and compare RNN, LSTM, and BiLSTM models for sentiment classification
- Preprocess text data with advanced tokenization and sequence handling
- Train multiple neural network architectures with proper validation
- Implement ensemble methods for improved performance
- Conduct comprehensive model evaluation and comparison

## 🧠 Model Architectures

### 1. RNN (Recurrent Neural Network)

RNNClassifier(
  (embedding): Embedding(VOCAB_SIZE, 50)
  (rnn): RNN(50, 32, batch_first=True, dropout=0.1)
  (fc): Linear(32, 1)
)

### 2. LSTM (Long Short-Term Memory)
LSTMClassifier(
  (embedding): Embedding(VOCAB_SIZE, 50)
  (lstm): LSTM(50, 32, batch_first=True, dropout=0.1)
  (fc): Linear(32, 1)
)
### 3. BiLSTM (Bidirectional LSTM)
BiLSTMClassifier(
  (embedding): Embedding(VOCAB_SIZE, 50)
  (lstm): LSTM(50, 32, batch_first=True, dropout=0.1, bidirectional=True)
  (fc): Linear(64, 1)  # 64 = 32*2 for bidirectional
)
## 📊 Dataset

### Dataset Information
- **Source**: Custom reviews dataset (`reviews.csv`)
- **Total Samples**: [Number of reviews in your dataset]
- **Classes**: 2 (Binary Sentiment Classification)
- **Features**: Text reviews with sentiment labels

### Data Structure
The dataset contains the following columns:
- **review**: Raw text of the customer review
- **sentiment**: Sentiment label (positive/negative)

### Sample Data
| review | sentiment |
|--------|-----------|
| "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence..." | positive |
| "Terrible movie, waste of time and money. Poor acting and boring storyline." | negative |

### Class Distribution
- **Positive Reviews**: [Count] samples
- **Negative Reviews**: [Count] samples

### Data Preprocessing Pipeline

#### 1. Text Cleaning
- **HTML Tag Removal**: Strip HTML tags (`<br />`, `<p>`, etc.)
- **Lowercase Conversion**: Convert all text to lowercase
- **URL Removal**: Remove web links and URLs
- **Punctuation Handling**: Clean special characters while preserving basic punctuation
- **Whitespace Normalization**: Remove extra spaces and normalize whitespace

#### 2. Tokenization & Vocabulary
- **Tokenizer**: NLTK `word_tokenize`
- **Vocabulary Size**: ~30,000 words
- **Special Tokens**:
  - `<PAD>`: Padding token (index 0)
  - `<UNK>`: Unknown word token (index 1)
- **Minimum Frequency**: Words appearing less than 2 times are mapped to `<UNK>`

#### 3. Sequence Processing
- **Maximum Sequence Length**: 200 tokens
- **Padding**: Shorter sequences padded with `<PAD>`
- **Truncation**: Longer sequences truncated to 200 tokens
- **Sequence Lengths**: Tracked for packed sequence processing

#### 4. Data Splitting
- **Training Set**: 60% of data
- **Validation Set**: 20% of data  
- **Test Set**: 20% of data
- **Stratified Sampling**: Maintains class distribution across splits

### Data Loaders
- **Batch Size**: 64 samples
- **Shuffling**: Enabled for training, disabled for validation/test
- **Device Optimization**: Automatic transfer to GPU when available

### Sentiment Encoding
| Original Label | Encoded Value | Tensor Representation |
|---------------|---------------|---------------------|
| positive | 1 | torch.tensor([1]) |
| negative | 0 | torch.tensor([0]) |

### Dataset Statistics
- **Average Review Length**: [Number] tokens
- **Vocabulary Coverage**: [Percentage]% of tokens in vocabulary
- **Class Balance**: [Percentage]% positive, [Percentage]% negative
- **Sequence Length Distribution**: Normal/Gaussian distribution around mean length
