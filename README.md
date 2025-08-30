# Kaizen Engineering Document Assistant Suite

This repository contains a suite of intelligent applications built with Streamlit and Python for advanced document analysis. These tools leverage the Together AI API to provide capabilities like interactive chat, document comparison, and robust question-answering for various file formats, including PDFs, DOCX, and Excel files.

![Kaizen Engineering Document Assistant UI](https://storage.googleapis.com/maker-media-tool-store/b39d1b64-89c0-4560-936d-96695679ac62/Screenshot%202025-08-30%20at%2010.29.21%E2%80%AFAM.png)
*(Above: A screenshot of the `main.py` application interface)*

***

## ⚙️ Tech Stack

* **Framework:** Streamlit
* **Language:** Python
* **Core Libraries:** LangChain, PyMuPDF (fitz), Pandas, python-docx
* **AI/LLM API:** Together AI (`meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo`)

***

## 📂 Code Files Explained

This project consists of three distinct Streamlit applications and a requirements file. *(Note: `main.py` and `chatbot.py` are identical; `main.py` should be considered the primary entry point for the main application).*

### 1. `main.py` (Kaizen Engineering Document Assistant)

[cite_start]This is the central application of the suite, providing a versatile, chat-based interface for interacting with a database of documents[cite: 1].

**Key Responsibilities & Logic:**
* [cite_start]**Automatic Database Indexing:** On startup, the app automatically scans a `data` directory for new documents[cite: 1]. [cite_start]It processes them in the background using a `ThreadPoolExecutor` for efficiency[cite: 1].
* [cite_start]**Multi-Format Text Extraction:** It can extract text from a wide variety of file types (`pdf`, `xlsx`, `docx`, `csv`, etc.)[cite: 1]. [cite_start]For PDFs, it uses a powerful OCR-based approach: each page is converted to an image and sent to a vision model (`Llama-3.2-90B-Vision`) to transcribe the text accurately[cite: 1].
* [cite_start]**Metadata Generation & Caching:** After extracting text, it uses the LLM to generate key metadata (like `work_order`, `client`, and `summary`) for each document[cite: 1]. [cite_start]The extracted text is saved to an `ocr_cache` directory to avoid reprocessing, and the metadata is stored in an `index.json` file for quick searching[cite: 1].
* [cite_start]**Intelligent Chat Logic:** The chatbot determines the user's intent (`general_question`, `find_similar`, `compare_documents`)[cite: 1].
    * If a user asks a **general question**, it first searches the context of the currently uploaded file. [cite_start]If the answer isn't found, it expands the search to the entire database, using the indexed summaries to find relevant documents before synthesizing a comprehensive answer[cite: 1].
    * [cite_start]If the user wants to **find similar documents**, it uses the `work_order` number from the uploaded file's metadata to find matches in the database[cite: 1].

### 2. `concomp.py` (Intelligent PDF Comparator)

This is a specialized tool designed for a single purpose: to find the best match for an uploaded PDF within a database and generate a detailed comparison.

**Key Responsibilities & Logic:**
* **Manual Database Building:** Unlike `main.py`, this app requires the user to manually trigger the database indexing via a button in the sidebar. It processes all PDFs in an `uploads` folder.
* **Best Match Identification:** When a user uploads a PDF, the app first analyzes it to generate a summary. It then sends this summary along with the summaries of all documents in the database to the LLM, asking it to identify the **single best match**.
* **Detailed Comparison Generation:** Once the best match is found, the application feeds the text of both the uploaded document and the matched document into the LLM with a detailed prompt. This prompt instructs the AI to act as an expert analyst and provide a high-level summary, a structured markdown table comparing specific attributes, and a final conclusion.

### 3. `langchain.py` (Kaizen Q&A with LangChain)

This application offers a more streamlined and visually styled Q&A experience built on the **LangChain framework**. It is focused purely on asking questions about one or more documents uploaded in a single session.

**Key Responsibilities & Logic:**
* **Multiple Input Methods:** It supports uploading files (`pdf`, `zip`, images), fetching documents from a URL, or directly pasting text.
* **LangChain Integration:** This is the key differentiator. It uses LangChain's `CharacterTextSplitter` to break the extracted document text into manageable chunks. It then loads these chunks into a `load_qa_chain` (specifically the "stuff" chain type), which provides a robust and standardized way to perform question-answering over the document context.
* **Modern UI:** This script uses custom CSS to deliver a more polished and professional user interface, including a styled header, status badges, and formatted Q&A sections.
* **API Abstraction:** It initializes the `ChatOpenAI` class from LangChain, configured to use the Together AI API endpoint. This abstracts away the direct `requests.post` calls seen in the other scripts.

### 4. `requirements.txt`

This file is crucial for setting up the project's environment. It lists all the necessary Python libraries that need to be installed for the applications to run correctly.

* **`streamlit`**: The core framework for building the web UI.
* **`requests`**: For making HTTP requests to the Together AI API.
* **`PyMuPDF`**: Used to open and process PDF files, primarily for converting pages to images for OCR.
* **`Pillow`**: A library for opening, manipulating, and saving many different image file formats.
* **`python-dotenv`**: To load environment variables (like API keys) from a `.env` file.
* **`python-docx`**: For extracting text from Microsoft Word (.docx) documents.
* **`pandas`** & **`openpyxl`**: Used for reading and extracting data from Excel files (.xlsx, .xls) and CSVs.

***

## 🚀 Setup and Usage

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set Up API Key:**
    Create a file named `.env` in the root of the project directory and add your Together AI API key:
    ```
    TOGETHER_API_KEY="your_api_key_here"
    ```

### Running the Applications

You can run any of the three applications using the `streamlit run` command.

* To run the main Document Assistant:
    ```bash
    streamlit run main.py
    ```
* To run the PDF Comparator:
    ```bash
    streamlit run concomp.py
    ```
* To run the LangChain Q&A app:
    ```bash
    streamlit run langchain.py
    ```
