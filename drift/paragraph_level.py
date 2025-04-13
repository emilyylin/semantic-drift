import matplotlib.pyplot as plt
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def split_paragraphs (text):

    # 2 newlines after sentence boundaries to split
    paragraphs = re.split(r'\n\s*\n', text.strip()) 

    return [p.strip() for p in paragraphs if p.strip()]

# calculate cosine sim from sentence to sentence
def cosine_drift(embedding):
    return [util.cos_sim(embedding[i], embedding[i+1]).item() for i in range(len(embedding)-1)]

# get text files
with open("../assets/text/books_human.txt", "r", encoding="utf-8") as f:
    human_text = f.read()
with open("../assets/text/books_ai.txt", "r", encoding="utf-8") as f:
    ai_text = f.read()

# process text files into paragraphs
human_paragraphs = split_paragraphs(human_text)
ai_paragraphs = split_paragraphs(ai_text)

# get sentence embeddings, cosine similarities
model = SentenceTransformer("all-MiniLM-L6-v2")
human_paragraph_embeddings = model.encode(human_paragraphs)
ai_paragraph_embeddings = model.encode(ai_paragraphs)

# stack to ensure same dimensions
pca = PCA (n_components=2)
all = np.vstack((ai_paragraph_embeddings,human_paragraph_embeddings))
reduced = pca.fit_transform(all)

# split back
ai_reduced = reduced [:len(ai_paragraph_embeddings)]
human_reduced = reduced[len(human_paragraph_embeddings):]

ai_reduced = reduced [:len(ai_paragraph_embeddings)]
human_reduced = reduced[len(human_paragraph_embeddings):]

# PLOT 
# k means clustering
k = 5
kmeans_human = KMeans(n_clusters=k, random_state=0).fit(human_reduced)
kmeans_ai = KMeans(n_clusters=k, random_state=0).fit(ai_reduced)

plt.figure(figsize=(10, 6))

plt.scatter(human_reduced[:, 0], human_reduced[:, 1], c=kmeans_human.labels_, cmap='Blues', label='Human', alpha=0.6)
plt.scatter(ai_reduced[:, 0], ai_reduced[:, 1], c=kmeans_ai.labels_, cmap='Reds', label='AI', alpha=0.6)

plt.scatter(kmeans_human.cluster_centers_[:, 0], kmeans_human.cluster_centers_[:, 1], color='blue', marker='X', s=100)
plt.scatter(kmeans_ai.cluster_centers_[:, 0], kmeans_ai.cluster_centers_[:, 1], color='red', marker='X', s=100)

plt.legend()
plt.title("KMeans Clustering of Semantic Space (Human vs AI)")
plt.grid(True)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
plt.show()