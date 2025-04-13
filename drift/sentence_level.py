import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def split_sentences (text):
    # space after sentence boundaries to split
    sentences = re.split(r'(?<=[.!?])\s+', text.strip()) 

    return [s.strip() for s in sentences if s.strip()]

# calculate cosine sim from sentence to sentence
def cosine_drift(embedding):
    return [util.cos_sim(embedding[i], embedding[i+1]).item() for i in range(len(embedding)-1)]

# return vector of drift differences
def cosine_delta(drift):
  return [abs(drift[i+1] - drift[i]) for i in range(len(drift) - 1)]

def smooth(values, window=10):
    return np.convolve(values, np.ones(window)/window, mode='valid')

# get text files
with open("../assets/text/books_human.txt", "r", encoding="utf-8") as f:
    human_text = f.read()
with open("../assets/text/books_ai.txt", "r", encoding="utf-8") as f:
    ai_text = f.read()

# process text files into sentences
human_sentences = split_sentences(human_text)
ai_sentences = split_sentences(ai_text)

# get sentence embeddings, cosine similarities
model = SentenceTransformer("all-MiniLM-L6-v2")
human_sentence_embeddings = model.encode(human_sentences)
ai_sentence_embeddings = model.encode(ai_sentences)

# stack to ensure same dimensions
pca = PCA (n_components=2)
all = np.vstack((ai_sentence_embeddings,human_sentence_embeddings))
reduced = pca.fit_transform(all)

# split back
ai_reduced = reduced [:len(ai_sentence_embeddings)]
human_reduced = reduced[len(human_sentence_embeddings):]

# plot - sentence to sentence cosine drift
human_drift=cosine_drift(human_sentence_embeddings)
ai_drift=cosine_drift(ai_sentence_embeddings)

plt.plot(smooth(human_drift), label="Human Drift", marker='o', alpha=0.6)
plt.plot(smooth(ai_drift), label="AI Drift", marker='o', alpha=0.6)
plt.title("Sentence-to-Sentence Cosine Drift")
plt.xlabel("Sentence Index")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# of consecutive deltas
human_delta = cosine_delta(human_drift)
ai_delta = cosine_delta(ai_drift)

plt.plot(smooth(human_delta), label="Human Drift Delta", marker='o', alpha=0.6)
plt.plot(smooth(ai_delta), label="AI Drift Deltas", marker='o', alpha=0.6)
plt.title("Sentence-to-Sentence Cosine Drift Deltas")
plt.xlabel("Sentence Index")
plt.ylabel("Cosine Similarity Delta")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# mapping out paths
# human path
for i in range(len(human_reduced) - 1):
    plt.arrow(human_reduced[i, 0], human_reduced[i, 1],
              human_reduced[i+1, 0] - human_reduced[i, 0],
              human_reduced[i+1, 1] - human_reduced[i, 1],
              color='blue', alpha=0.4, head_width=0.005, length_includes_head=True)

# ai path
for i in range(len(ai_reduced) - 1):
    plt.arrow(ai_reduced[i, 0], ai_reduced[i, 1],
              ai_reduced[i+1, 0] - ai_reduced[i, 0],
              ai_reduced[i+1, 1] - ai_reduced[i, 1],
              color='red', alpha=0.4, head_width=0.005, length_includes_head=True)

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