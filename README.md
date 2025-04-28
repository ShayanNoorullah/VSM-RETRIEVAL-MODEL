# VSM (Vector Space Model) Document Retrieval System

A powerful and interactive document retrieval system that implements the Vector Space Model (VSM) for efficient text search and retrieval. This system provides both a command-line interface and a graphical user interface for searching through document collections.

## 🌟 Features

- **Advanced Text Processing**
  - Tokenization and lemmatization
  - Stopword removal
  - Punctuation handling
  - Case normalization

- **Sophisticated Search Capabilities**
  - Vector Space Model implementation
  - TF-IDF weighting
  - Cosine similarity scoring
  - Phrase query support
  - Positional indexing

- **User-Friendly Interface**
  - Modern GUI with tkinter
  - Real-time search results
  - Search history tracking
  - Document frequency visualization
  - Interactive result display

- **Performance Optimizations**
  - Efficient index storage using pickle
  - Cached document processing
  - Optimized similarity calculations
  - Fast query processing

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python packages:
  ```
  numpy
  pandas
  scikit-learn
  nltk
  tkinter
  ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/VSM-RETRIEVAL-MODEL.git
   cd VSM-RETRIEVAL-MODEL
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK data:
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

### Project Structure

```
VSM-RETRIEVAL-MODEL/
├── Abstracts/              # Document collection
├── VSMModelGui.py         # Main application file
├── Stopword-List.txt      # Stopwords for text processing
├── documentFrequencies.xlsx # Document frequency statistics
└── README.md              # This documentation
```

## 💻 Usage

### Running the Application

1. Start the GUI application:
   ```bash
   python VSMModelGui.py
   ```

2. The main interface will appear with:
   - Search query input field
   - Phrase query toggle
   - Search button
   - Results display area

### Performing Searches

1. **Basic Search**
   - Enter your search query in the input field
   - Click "Search" or press Enter
   - Results will be displayed with relevance scores

2. **Phrase Search**
   - Check the "Phrase Query" checkbox
   - Enter your phrase query
   - Click "Search"
   - Results will show documents containing the exact phrase

### Understanding Results

- Results are ranked by relevance score
- Each result shows:
  - Document ID
  - Relevance score
  - Execution time
  - Document content preview

## 🔧 Technical Details

### Vector Space Model Implementation

The system implements a Vector Space Model with:
- TF-IDF weighting
- Cosine similarity
- Positional indexing
- Document frequency tracking

### Index Management

- Indexes are automatically created on first run
- Stored in `vsmIndex.pkl`
- Automatically rebuilt if corrupted

## 📊 Performance

- Fast query processing
- Efficient index storage
- Optimized similarity calculations
- Cached document processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 👥 Authors
- Shayan: {https://github.com/ShayanNoorullah}

## 🙏 Acknowledgments

- NLTK for natural language processing
- scikit-learn for similarity calculations
- tkinter for GUI implementation 