

---

# 📧 Spam Email Detector 📊

A Python-based project to detect spam emails using text processing, graph analysis, and machine learning techniques. This project utilizes co-occurrence graphs and centrality metrics to classify emails as **spam** or **ham**.

---

## 🚀 Features

- 🛠️ **Text Preprocessing**: Tokenizes email content and removes stopwords.
- 📈 **Graph-Based Analysis**: Creates co-occurrence graphs from email content and computes:
  - Average centrality
  - Maximum centrality
  - Graph density
  - Number of connected components
  - Clustering coefficient
- 🤖 **Machine Learning**: Uses a Random Forest classifier to classify emails as spam or ham.
- 📊 **Visualizations**:
  - Co-occurrence graph visualization.
  - Centrality metric distribution plots.

---

## 📂 Dataset Requirements

- The dataset should be in `.csv` format and include the following columns:
  - **`text`**: The content of the email.
  - **`label`**: The classification of the email (`spam` or `ham`).

Example row:
```csv
text,label
"This is an important update about your account.",ham
"Win $1000 now! Click here!",spam
```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Baasim-Ahmed/Spam-Email-Detector.git
   cd Spam-Email-Detector
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

---

## 🖥️ Usage

1. Place your dataset in the root directory (e.g., `emails.csv`).
2. Run the script:
   ```bash
   python spam_email_detector.py
   ```
3. View the results:
   - Model accuracy will be displayed in the console.
   - A boxplot for centrality distribution will be generated.
   - A sample co-occurrence graph will be visualized.

---

## 📝 Code Highlights

### Text Preprocessing
```python
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens
```

### Co-occurrence Graph
```python
def build_cooccurrence_graph(tokens):
    graph = nx.Graph()
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            if tokens[i] != tokens[j]:
                graph.add_edge(tokens[i], tokens[j])
    return graph
```

---


## 📦 Dependencies

- **pandas**
- **nltk**
- **networkx**
- **scikit-learn**
- **matplotlib**
- **seaborn**

Install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## 🛡️ License

This project is licensed under the MIT License.

---

## ✨ Acknowledgments

- Built with ❤️ by **[Baasim Ahmed](https://github.com/Baasim-Ahmed)**.
- Thanks to the Python and open-source community for their amazing tools!

---

