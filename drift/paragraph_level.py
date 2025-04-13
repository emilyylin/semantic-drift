import matplotlib.pyplot as plt
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from mpl_toolkits.mplot3d import Axes3D

def split_paragraphs (text):

    # 2 newlines after sentence boundaries to split
    paragraphs = re.split(r'\n\s*\n', text.strip()) 

    return [p.strip() for p in paragraphs if p.strip()]

# calculate cosine sim from sentence to sentence
def cosine_drift(embedding):
    return [util.cos_sim(embedding[i], embedding[i+1]).item() for i in range(len(embedding)-1)]

# get text files
with open("assets/books_human.txt", "r", encoding="utf-8") as f:
    human_text = f.read()
with open("assets/books_ai.txt", "r", encoding="utf-8") as f:
    ai_text = f.read()

# process text files into paragraphs
human_paragraphs = split_paragraphs(human_text)
ai_paragraphs = split_paragraphs(ai_text)

