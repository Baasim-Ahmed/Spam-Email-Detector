# Import necessary libraries
import pandas as pd
import nltk
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset (replace 'emails.csv' with the path to your dataset)
# Dataset must have columns 'text' (email content) and 'label' (spam/ham)
data = pd.read_csv('emails.csv')  # Example: 'label' column contains 'spam' or 'ham'
data['label'] = data['label'].map({'spam': 1, 'ham': 0})  # Convert labels to binary (spam=1, ham=0)

# Define a preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    stop_words = set(stopwords.words('english'))  # Get English stopwords
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove stopwords and non-alphanumeric
    return tokens

# Apply preprocessing
data['tokens'] = data['text'].apply(preprocess_text)

# Function to build a co-occurrence graph
def build_cooccurrence_graph(tokens):
    if not tokens or len(tokens) < 2:  # Handle empty or small token lists
        return nx.Graph()  # Return an empty graph
    
    graph = nx.Graph()
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            if tokens[i] != tokens[j]:
                graph.add_edge(tokens[i], tokens[j])
    return graph

# Generate a graph for each email
data['graph'] = data['tokens'].apply(build_cooccurrence_graph)

# Function to compute centrality metrics
def compute_centrality_metrics(graph):
    if graph.number_of_nodes() < 2:  # Handle small graphs
        return {
            'avg_centrality': 0,
            'max_centrality': 0,
        }

    centrality = nx.degree_centrality(graph)
    return {
        'avg_centrality': sum(centrality.values()) / len(centrality),
        'max_centrality': max(centrality.values()),
    }

# Apply centrality computation
data['centrality'] = data['graph'].apply(compute_centrality_metrics)
data['avg_centrality'] = data['centrality'].apply(lambda x: x['avg_centrality'])
data['max_centrality'] = data['centrality'].apply(lambda x: x['max_centrality'])

# Function to compute advanced graph metrics
def compute_advanced_metrics(graph):
    if graph.number_of_nodes() < 2:  # Handle small or empty graphs
        return {
            'density': 0,
            'num_components': 0,
            'clustering_coefficient': 0,
        }
    
    return {
        'density': nx.density(graph),
        'num_components': nx.number_connected_components(graph),
        'clustering_coefficient': nx.average_clustering(graph),
    }

# Apply advanced metrics computation
data['advanced_metrics'] = data['graph'].apply(compute_advanced_metrics)
data['density'] = data['advanced_metrics'].apply(lambda x: x['density'])
data['num_components'] = data['advanced_metrics'].apply(lambda x: x['num_components'])
data['clustering_coefficient'] = data['advanced_metrics'].apply(lambda x: x['clustering_coefficient'])

# Debugging step: Check graph structures
for idx, graph in enumerate(data['graph']):
    logging.info(f"Processing graph at index {idx}")
    logging.info(f"Nodes: {list(graph.nodes())}")
    logging.info(f"Edges: {list(graph.edges())}")
    logging.info(f"Graph Type: {type(graph)}")

# Function to visualize a sample graph
def visualize_graph(graph, title):
    if graph.number_of_nodes() == 0:  # Handle empty graphs gracefully
        print("Empty graph. No visualization available.")
        return
    
    plt.figure(figsize=(8, 6))
    nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

# Visualize a sample email graph (e.g., the first email in the dataset)
visualize_graph(data['graph'][0], "Sample Email Graph")

# Extract features (centrality metrics) and labels
features = data[['avg_centrality', 'max_centrality', 'density', 'num_components', 'clustering_coefficient']]
labels = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Display metrics for insights
print("Centrality Metrics for Spam and Ham Emails:")
print(data.groupby('label')[['avg_centrality', 'max_centrality', 'density', 'clustering_coefficient']].mean())

# Visualize centrality distribution
sns.boxplot(x=data['label'], y=data['avg_centrality'])
plt.title("Average Centrality Distribution (0=Ham, 1=Spam)")
plt.show()
