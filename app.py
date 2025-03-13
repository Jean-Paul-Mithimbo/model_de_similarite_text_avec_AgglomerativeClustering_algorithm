import os
from flask import Flask, request, render_template, redirect, url_for
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import io
import base64

# Initialiser l'application Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'document/'

# S'assurer que le dossier de téléchargement existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Télécharger les stop words en français
nltk.download('stopwords')
french_stop_words = stopwords.words('french')

# Fonction pour extraire le texte d'un fichier PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Route pour télécharger des fichiers et lancer le clustering
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            return redirect(request.url)
        
        # Sauvegarder les nouveaux fichiers téléchargés
        for file in files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

        return redirect(url_for('verify_documents'))

    return render_template('upload.html')

# Route pour vérifier les documents existants dans le dossier
@app.route('/verify', methods=['GET'])
def verify_documents():
    # Extraire le texte de tous les fichiers PDF dans le dossier
    documents = []
    filenames = []
    for pdf_name in os.listdir(app.config['UPLOAD_FOLDER']):
        if pdf_name.endswith('.pdf'):
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_name)
            text = extract_text_from_pdf(pdf_path)
            documents.append(text)
            filenames.append(pdf_name)

    # Vérifier s'il y a au moins deux documents
    if len(documents) < 2:
        return render_template('error.html', message="Veuillez télécharger au moins deux fichiers PDF pour le clustering.")

    # Prétraitement et vectorisation du texte
    vectorizer = TfidfVectorizer(stop_words=french_stop_words)
    X = vectorizer.fit_transform(documents)

    # Application de l'Agglomerative Clustering
    agglo = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    agglo.fit(X.toarray())

    # Génération du dendrogramme
    Z = linkage(X.toarray(), 'ward')
    plt.figure(figsize=(15, 10))  # Ajuster la taille de la figure
    dendrogram(Z, labels=filenames, leaf_rotation=90)  # Rotation des étiquettes
    plt.title('Dendrogramme des documents')
    plt.xlabel('Documents')
    plt.ylabel('Distance')

    # Sauvegarder le graphique dans un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return render_template('result.html', img_base64=img_base64)

if __name__ == '__main__':
    app.run(debug=True)