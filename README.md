# 📰 Fake News Detection using Machine Learning

A Machine Learning and Natural Language Processing (NLP) project that detects whether a news article is **Real** or **Fake** using trained classification models.

This project uses text preprocessing, TF-IDF vectorization, and Machine Learning algorithms to classify news content efficiently through a simple Streamlit web application.

---

# 📌 Project Overview

Fake news has become a major issue in digital media and social networking platforms.  
This project aims to automatically identify and classify fake news articles using Machine Learning techniques.

The system performs:

- Text preprocessing
- Data cleaning
- Feature extraction
- Model training
- Fake/Real news prediction

---

# 🚀 Features

- 📰 Fake News Classification
- 🤖 Machine Learning Prediction
- 🔍 NLP Text Processing
- 📊 TF-IDF Vectorization
- 🌐 Streamlit Web Interface
- 💾 Saved Trained Model
- ⚡ Real-time Prediction

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLP
- TF-IDF Vectorizer
- Streamlit
- Pickle

---

# 📂 Project Structure

```bash
fake_news_detection_Predictive3/
│
├── app.py
├── requirements.txt
├── model.pkl
├── vectorizer.pkl
├── fake-news-detection.ipynb
├── README.md
│
└── dataset/
    ├── Fake.csv
    └── True.csv
```

---

# 📊 Dataset

The dataset contains two CSV files:

| File Name | Description |
|------------|-------------|
| Fake.csv | Fake news articles |
| True.csv | Real news articles |

Dataset includes:

- News title
- News content
- Labels for classification

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/krithiika-s/fake_news_detection_Predictive3.git
```

---

## 2️⃣ Navigate to Project Folder

```bash
cd fake_news_detection_Predictive3
```

---

## 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

After running the command, open the local URL displayed in the terminal.

Example:

```bash
http://localhost:8501
```

---

# 🔍 How the Project Works

## Step 1: Data Preprocessing

- Remove unwanted characters
- Convert text to lowercase
- Remove stopwords
- Clean text data

## Step 2: Feature Extraction

TF-IDF Vectorization converts textual data into numerical format.

## Step 3: Model Training

Machine Learning models are trained using the processed dataset.

## Step 4: Prediction

The model predicts whether the news is:

- REAL
- FAKE

---

# 📈 Workflow

```text
News Input
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Machine Learning Model
    ↓
Prediction Result
```

---

# 🖥 Example Usage

1. Open the Streamlit application
2. Enter a news article or headline
3. Click the Predict button
4. View the prediction result

---

# 📚 Future Enhancements

- Add Deep Learning models
- Improve accuracy
- Add multilingual support
- Deploy using Streamlit Cloud
- Add live news verification APIs

---




---

# 📄 License

This project is developed for educational purposes.

---

# 👩‍💻 Author

Developed by **Krithiika**

GitHub Repository:  
https://github.com/krithiika-s/fake_news_detection_Predictive3

---

