# import fitz  # PyMuPDF
# import os
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import nltk
# from nltk.corpus import stopwords

# # Download French stop words
# nltk.download('stopwords')
# french_stop_words = stopwords.words('french')

# # Chemin vers le dossier contenant les PDF
# pdf_folder = 'document/'

# # Fonction pour extraire le texte des fichiers PDF
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         text += page.get_text()
#     return text

# # Extraction du texte de tous les PDF dans le dossier
# documents = []
# for pdf_name in os.listdir(pdf_folder):
#     if pdf_name.endswith('.pdf'):
#         pdf_path = os.path.join(pdf_folder, pdf_name)
#         text = extract_text_from_pdf(pdf_path)
#         documents.append(text)

# # Prétraitement et vectorisation des textes
# vectorizer = TfidfVectorizer(stop_words=french_stop_words)
# X = vectorizer.fit_transform(documents)

# # Application de K-means avec K=3 (ou ajustez selon vos besoins)
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# labels = kmeans.labels_

# # Affichage des résultats
# for i, label in enumerate(labels):
#     print(f'Document: {os.listdir(pdf_folder)[i]} - Cluster: {label}')



import fitz  # PyMuPDF
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

# Download French stop words
nltk.download('stopwords')
french_stop_words = stopwords.words('french')

# Chemin vers le dossier contenant les PDF
pdf_folder = 'document/'

# Fonction pour extraire le texte des fichiers PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Extraction du texte de tous les PDF dans le dossier
documents = []
for pdf_name in os.listdir(pdf_folder):
    if pdf_name.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, pdf_name)
        text = extract_text_from_pdf(pdf_path)
        documents.append(text)

# Prétraitement et vectorisation des textes
vectorizer = TfidfVectorizer(stop_words=french_stop_words)
X = vectorizer.fit_transform(documents)

# Application de Agglomerative Clustering
agglo = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
agglo.fit(X.toarray())

# Génération du dendrogramme
Z = linkage(X.toarray(), 'ward')
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=[os.listdir(pdf_folder)[i] for i in range(len(documents))])
plt.title('Dendrogramme des documents')
plt.xlabel('Documents')
plt.ylabel('Distance')
plt.show()
