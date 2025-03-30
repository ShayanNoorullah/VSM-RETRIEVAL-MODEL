#22k-4148 SHAYAN 6-H
import os
import string
import pickle
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from math import log
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

# DOWNLOADING REQUIRED NLTK MODULES FOR THE QUERY PROCESSING:
nltk.download('wordnet')
nltk.download('punkt')

# PICKLE SERIALIZATION FACTORY FUNCTIONS FOR USING PICKLE LIBRARY EFFICIENTLY:
def listFactory():
    return []
def docPositionFactory():
    return defaultdict(listFactory)
def intFactory():
    return 0

# CLASS FOR PERFORMING REQUIRED VSM MODEL TASKS:
class VectorSpaceModel:
    def __init__(self, datasetPath, stopwordsPath, indexPath='vsmIndex.pkl'):
        # INITIALISATION OF VSM WITH DATASET AND STOPWORDS:
        self.datasetPath = datasetPath
        self.stopwords = self.loadStopwords(stopwordsPath)
        self.indexPath = indexPath
        self.documents = {}
        self.index = defaultdict(dict)
        self.positionalIndex = defaultdict(docPositionFactory)
        self.docFreq = defaultdict(int)
        self.tfIdfMatrix = None
        self.terms = []
        
        try:
            if os.path.exists(indexPath) and os.path.getsize(indexPath) > 0:
                self.loadIndex()
                print("Loaded existing index")
                return
        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            print(f"Corrupted index file: {e}, rebuilding...")
            if os.path.exists(indexPath):
                os.remove(indexPath)

        self.documents = self.loadDocuments()
        self.preprocessDocuments()
        self.buildIndex()
        self.saveIndex()
        print("Created new index")

    # fUNTION FOR LOADING STOPWORDS FROM THE SPECIFIED FILE:
    def loadStopwords(self, path):
        with open(path, encoding="utf-8", errors='ignore') as f:
            return set(f.read().lower().split())

    # FUNCTION TO LOAD DOCUMENT FROM THE SPECIFIED ABSTRACTS FOLDER:
    def loadDocuments(self):
        
        docs = {}
        for filename in sorted(os.listdir(self.datasetPath), 
                            key=lambda x: int(x.split('.')[0])):
            with open(os.path.join(self.datasetPath, filename), 'r', 
                    encoding='utf-8', errors='ignore') as file:
                docs[int(filename.split('.')[0])] = file.read()
        return docs

    # TEXT PREPROCESSING BY TOKENISATION, INDEXING AND REMOVING STOPWORDS:
    def preprocessText(self, text):
        lemmatizer = WordNetLemmatizer()
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        return [lemmatizer.lemmatize(word) for word in tokens 
                if word.isalnum() and word not in self.stopwords]

    # DOCUMENTS PREPROCESSING IN THE CORPUS: 
    def preprocessDocuments(self):
        for docId in list(self.documents.keys()):
            self.documents[docId] = self.preprocessText(self.documents[docId])

    # CREATING TF-IDF MATRIX AND POSITIONAL INDEX:
    def buildIndex(self):
        totalDocs = len(self.documents)
        termDocMatrix = defaultdict(lambda: defaultdict(intFactory))

        # CREATING POSITIONAL INDEX:
        for docId, tokens in self.documents.items():
            termCounts = Counter(tokens)
            for pos, term in enumerate(tokens):
                self.positionalIndex[term][docId].append(pos)
                termDocMatrix[term][docId] = termCounts[term]
                self.docFreq[term] += 1

        # CRATING TF-IDF MATRIX:
        self.terms = sorted(termDocMatrix.keys())
        self.tfIdfMatrix = np.zeros((len(self.terms), totalDocs))
        for termIdx, term in enumerate(self.terms):
            df = self.docFreq[term]
            idf = log(totalDocs / df) if df > 0 else 0
            for docId, tf in termDocMatrix[term].items():
                docIdx = list(self.documents.keys()).index(docId)
                self.tfIdfMatrix[termIdx, docIdx] = tf * idf

    # PROCESSING PHRASE QUERIES:
    def phraseQuery(self, queryTokens):
        if not queryTokens:
            return []
        commonDocs = set(self.positionalIndex.get(queryTokens[0], {}).keys())
        for term in queryTokens[1:]:
            commonDocs &= set(self.positionalIndex.get(term, {}).keys())
        validDocs = []
        for docId in commonDocs:
            positions = [self.positionalIndex[term][docId] for term in queryTokens]
            
            # PERFORMING POSITIONAL VERIFICATION:
            firstTermPos = positions[0]
            for pos in firstTermPos:
                valid = True
                for i in range(1, len(positions)):
                    if (pos + i) not in positions[i]:
                        valid = False
                        break
                if valid:
                    validDocs.append(docId)
                    break
        return validDocs

    # PERFORMING QUERY PROCESSINGAND RETURNING MATCHED RESULTS SORTED BY DOCUMENT RANKING:
    def query(self, queryText, isPhraseQuery=False, threshold=0.05):
        startTime = time.time()
        queryTokens = self.preprocessText(queryText)
        if isPhraseQuery:
            results = self.phraseQuery(queryTokens)
            execTime = time.time() - startTime
            return results, execTime
        
        # VECTOR SPACE QP:
        queryVector = np.zeros(len(self.terms))
        termCounts = Counter(queryTokens)
        totalDocs = len(self.documents)

        for term, tf in termCounts.items():
            if term in self.terms:
                termIdx = self.terms.index(term)
                df = self.docFreq.get(term, 0)
                idf = log(totalDocs / df) if df > 0 else 0
                queryVector[termIdx] = tf * idf

        # COSINE SIMILARITY CALCULATIONS:
        docScores = cosine_similarity([queryVector], self.tfIdfMatrix.T)[0]
        rankedResults = sorted(
            [(docId, score) for docId, score in zip(self.documents.keys(), docScores) 
             if score >= threshold],
            key=lambda x: x[1], 
            reverse=True
        )
        docIds = [docId for docId, _ in rankedResults]
        execTime = time.time() - startTime
        return docIds, execTime

    # INDEX SAVING IN FILE:
    def saveIndex(self):
        tempPath = self.indexPath + ".tmp"
        try:
            with open(tempPath, 'wb') as f:
                pickle.dump({
                    'positionalIndex': self.positionalIndex,
                    'tfIdfMatrix': self.tfIdfMatrix,
                    'docFreq': self.docFreq,
                    'terms': self.terms,
                    'documents': self.documents
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # FILE REPLACEMENT FOR ATOMICITY:
            if os.path.exists(self.indexPath):
                os.remove(self.indexPath)
            os.rename(tempPath, self.indexPath)
        except Exception as e:
            print(f"Error saving index: {e}")
            if os.path.exists(tempPath):
                os.remove(tempPath)

    # LOAD INDEX DATE FROM FILE:
    def loadIndex(self):
        try:
            with open(self.indexPath, 'rb') as f:
                indexData = pickle.load(f)
            requiredKeys = {'positionalIndex', 'tfIdfMatrix', 
                           'docFreq', 'terms', 'documents'}
            if not all(key in indexData for key in requiredKeys):
                raise ValueError("Invalid index format")
                
            self.positionalIndex = indexData['positionalIndex']
            self.tfIdfMatrix = indexData['tfIdfMatrix']
            self.docFreq = indexData['docFreq']
            self.terms = indexData['terms']
            self.documents = indexData['documents']
        except Exception as e:
            print(f"Error loading index: {e}")
            raise

# CLASS FOR IMPLEMENTING GUI FOR VSM
class VSMGUI:
    def __init__(self, vsm):
        self.vsm = vsm
        self.root = tk.Tk()
        self.root.title("VSM Search Interface")
        self.root.geometry("800x600")
        self.searchHistory = []
        
        self.configureStyles()
        self.createWidgets()
        self.root.mainloop()

    # VISUAL STYLES ONFIGURATION USING TTK MODULE:
    def configureStyles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', padding=6, relief='flat',
                           background='#4CAF50', foreground='white')
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'),
                           foreground='#2c3e50')
        self.style.configure('Result.TFrame', background='#ecf0f1')

    # CREATING AND STRUCTURING GUI COMPONENTS:
    def createWidgets(self):
        mainFrame = ttk.Frame(self.root)
        mainFrame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(mainFrame, text="VSM Document Search", 
                 style='Header.TLabel').pack(pady=10)
        
        inputFrame = ttk.Frame(mainFrame)
        inputFrame.pack(fill='x', pady=5)
        
        ttk.Label(inputFrame, text="Search Query:").pack(side='left')
        self.queryEntry = ttk.Entry(inputFrame, width=60)
        self.queryEntry.pack(side='left', padx=10)
        
        self.phraseVar = tk.BooleanVar()
        ttk.Checkbutton(inputFrame, text="Phrase Query", 
                       variable=self.phraseVar).pack(side='left')

        btnFrame = ttk.Frame(mainFrame)
        btnFrame.pack(pady=10)
        actions = [
            ('Search', self.handleSearch, '#4CAF50'),
            ('Save Dictionary', self.saveDictionary, '#FF9800'),
            ('Show History', self.showHistory, '#2196F3'),
            ('Exit', self.root.destroy, '#F44336')
        ]
        
        for text, cmd, color in actions:
            btn = ttk.Button(btnFrame, text=text, command=cmd,
                            style='TButton')
            btn.pack(side='left', padx=5)
            self.style.configure(btn, background=color)

        resultFrame = ttk.Frame(mainFrame, style='Result.TFrame')
        resultFrame.pack(fill='both', expand=True)
        
        self.resultText = scrolledtext.ScrolledText(
            resultFrame, wrap=tk.WORD, font=('Consolas', 10),
            width=85, height=20
        )
        self.resultText.pack(padx=10, pady=10, fill='both', expand=True)

    # PROCESSING AND DISPLAYING THE SEARCH RESULTS IN GUI:
    def handleSearch(self):
        query = self.queryEntry.get().strip()
        if not query:
            messagebox.showwarning("Input Error", "Please enter a search query")
            return
        
        try:
            results, execTime = self.vsm.query(query, self.phraseVar.get())
            resultStr = f"Query: {query}\nExecution Time: {execTime:.4f}s\n"
            
            if results:
                resultStr += f"Found {len(results)} documents:\n" + ', '.join(map(str, results))
            else:
                resultStr += "No matching documents found"
                
            self.resultText.delete(1.0, tk.END)
            self.resultText.insert(tk.END, resultStr)
            self.searchHistory.append((query, results))
        except Exception as e:
            messagebox.showerror("Search Error", f"Error processing query: {str(e)}")

    # CREATING EXCEL FILE TO SAVE DICTIONARY EXPORTING THE TERM FREQUENCIES OF ALL TERMS IN THE DOCUMENTS IN CORPUS:
    def saveDictionary(self):
        try:
            df = pd.DataFrame.from_dict(self.vsm.docFreq, 
                                      orient='index', 
                                      columns=['Document Frequency'])
            df.to_excel("documentFrequencies.xlsx", index=True)
            messagebox.showinfo("Success", 
                             "Document frequencies saved to documentFrequencies.xlsx")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save dictionary: {str(e)}")

    # DISPLAYING SEARCH HISTORY:
    def showHistory(self):
        historyEntries = []
        for idx, (query, docs) in enumerate(self.searchHistory, 1):
            entry = [
                f"Query {idx}: {query}",
                f"Documents: {', '.join(map(str, docs)) if docs else 'None'}"
            ]
            historyEntries.append('\n'.join(entry))
        messagebox.showinfo("Search History", 
                          '\n\n'.join(historyEntries) if historyEntries 
                          else "No search history available")

# MAIN FUCNTION TO RUN THE VSM MODEL VIA GUI:
if __name__ == "__main__":
    datasetPath = 'Abstracts'
    stopwordsPath = 'Stopword-List.txt'
    vsmInstance = VectorSpaceModel(datasetPath, stopwordsPath)
    VSMGUI(vsmInstance)