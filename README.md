# Kaizen Engineering Document Assistant Suite

This repository contains a suite of intelligent applications built with Streamlit and Python for advanced document analysis. These tools leverage the Together AI API to provide capabilities like interactive chat, document comparison, and robust question-answering for various file formats, including PDFs, DOCX, and Excel files.
<img width="1710" height="854" alt="Screenshot 2025-08-30 at 10 29 21 AM" src="https://github.com/user-attachments/assets/ce50b7e1-3115-4d64-9b46-ef2ec64057ab" />

*(Above: A screenshot of the `main.py` application interface)*

-----

## ⚙️ Tech Stack

  * **Framework:** Streamlit
  * **Language:** Python
  * **Core Libraries:** LangChain, PyMuPDF (fitz), Pandas, python-docx
  * **AI/LLM API:** Together AI (`meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo`)

-----

## 📂 Code Files Explained

This project consists of three distinct Streamlit applications and a requirements file. *(Note: `main.py` and `chatbot.py` are identical; `main.py` should be considered the primary entry point for the main application).*

### 1\. `main.py` (Kaizen Engineering Document Assistant)

This is the central application of the suite, providing a versatile, chat-based interface for interacting with a database of documents.

**Key Responsibilities & Logic:**

  * **Automatic Database Indexing:** On startup, the app automatically scans a `data` directory for new documents. It processes them in the background using a `ThreadPoolExecutor` for efficiency.
  * **Multi-Format Text Extraction:** It can extract text from a wide variety of file types (`pdf`, `xlsx`, `docx`, `csv`, etc.). For PDFs, it uses a powerful OCR-based approach: each page is converted to an image and sent to a vision model (`Llama-3.2-90B-Vision`) to transcribe the text accurately.
  * **Metadata Generation & Caching:** After extracting text, it uses the LLM to generate key metadata (like `work_order`, `client`, and `summary`) for each document. The extracted text is saved to an `ocr_cache` directory to avoid reprocessing, and the metadata is stored in an `index.json` file for quick searching.
  * **Intelligent Chat Logic:** The chatbot determines the user's intent (`general_question`, `find_similar`, `compare_documents`).
      * If a user asks a **general question**, it first searches the context of the currently uploaded file. If the answer isn't found, it expands the search to the entire database, using the indexed summaries to find relevant documents before synthesizing a comprehensive answer.
      * If the user wants to **find similar documents**, it uses the `work_order` number from the uploaded file's metadata to find matches in the database.

### 2\. `concomp.py` (Intelligent PDF Comparator)

This is a specialized tool designed for a single purpose: to find the best match for an uploaded PDF within a database and generate a detailed comparison.

**Key Responsibilities & Logic:**

  * **Manual Database Building:** Unlike `main.py`, this app requires the user to manually trigger the database indexing via a button in the sidebar. It processes all PDFs in an `uploads` folder.
  * **Best Match Identification:** When a user uploads a PDF, the app first analyzes it to generate a summary. It then sends this summary along with the summaries of all documents in the database to the LLM, asking it to identify the **single best match**.
  * **Detailed Comparison Generation:** Once the best match is found, the application feeds the text of both the uploaded document and the matched document into the LLM with a detailed prompt. This prompt instructs the AI to act as an expert analyst and provide a high-level summary, a structured markdown table comparing specific attributes, and a final conclusion.

### 3\. `langchain.py` (Kaizen Q\&A with LangChain)

This application offers a more streamlined and visually styled Q\&A experience built on the **LangChain framework**. It is focused purely on asking questions about one or more documents uploaded in a single session.

**Key Responsibilities & Logic:**

  * **Multiple Input Methods:** It supports uploading files (`pdf`, `zip`, images), fetching documents from a URL, or directly pasting text.
  * **LangChain Integration:** This is the key differentiator. It uses LangChain's `CharacterTextSplitter` to break the extracted document text into manageable chunks. It then loads these chunks into a `load_qa_chain` (specifically the "stuff" chain type), which provides a robust and standardized way to perform question-answering over the document context.
  * **Modern UI:** This script uses custom CSS to deliver a more polished and professional user interface, including a styled header, status badges, and formatted Q\&A sections.
  * **API Abstraction:** It initializes the `ChatOpenAI` class from LangChain, configured to use the Together AI API endpoint. This abstracts away the direct `requests.post` calls seen in the other scripts.

### 4\. `requirements.txt`

This file is crucial for setting up the project's environment. [cite\_start]It lists all the necessary Python libraries that need to be installed for the applications to run correctly. [cite: 1]

  * [cite\_start]**`streamlit`**: The core framework for building the web UI. [cite: 1]
  * [cite\_start]**`requests`**: For making HTTP requests to the Together AI API. [cite: 1]
  * [cite\_start]**`PyMuPDF`**: Used to open and process PDF files, primarily for converting pages to images for OCR. [cite: 1]
  * [cite\_start]**`Pillow`**: A library for opening, manipulating, and saving many different image file formats. [cite: 1]
  * [cite\_start]**`python-dotenv`**: To load environment variables (like API keys) from a `.env` file. [cite: 1]
  * [cite\_start]**`python-docx`**: For extracting text from Microsoft Word (.docx) documents. [cite: 1]
  * [cite\_start]**`pandas`** & **`openpyxl`**: Used for reading and extracting data from Excel files (.xlsx, .xls) and CSVs. [cite: 1]

-----

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
  * To run the LangChain Q\&A app:
    ```bash
    streamlit run langchain.py
    ```
