import streamlit as st
import requests
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import json
import shutil
from dotenv import load_dotenv
import time
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Assistant",
    page_icon="🧠",
    layout="wide"
)

# --- Removed custom styling to allow Streamlit's default dark/light theme ---


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Outputs logs to the console/terminal for production monitoring
    ]
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Constants & Initialization ---
DATABASE_DIR = "Uploads"
OCR_CACHE_DIR = "ocr_cache"
INDEX_FILE = "index.json"
API_KEY = os.getenv("TOGETHER_API_KEY")
SUPPORTED_FILE_TYPES = ('pdf', 'xlsx', 'xls', 'docx', 'txt', 'csv', 'md')
MAX_API_RETRIES = 3
API_TIMEOUT = 180 # seconds

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your documents today?"}]
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = {}
if "comparison_target" not in st.session_state:
    st.session_state.comparison_target = None


# --- Core API & File Processing Functions ---

@st.cache_data(show_spinner=False)
def pdf_to_images(file_content: bytes) -> List[Image.Image]:
    """Converts PDF file content into a list of PIL Images."""
    images = []
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        for page_num, page in enumerate(pdf_document):
            try:
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                buf = io.BytesIO(img_bytes)
                images.append(Image.open(buf))
            except Exception as e:
                logger.error(f"Failed to process page {page_num} of a PDF: {e}")
    except Exception as e:
        logger.error(f"Could not open PDF file for image conversion: {e}")
    return images

def call_together_api(prompt: str, image_base64: Optional[str] = None, max_tokens: int = 8192) -> Optional[str]:
    """Generic function to call Together AI API with retries and error handling."""
    if not API_KEY:
        logger.critical("TOGETHER_API_KEY is not configured.")
        return None

    endpoint = 'https://api.together.xyz/v1/chat/completions'
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    content = [{"type": "text", "text": prompt}]
    if image_base64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})

    payload = {
        "model": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "max_tokens": max_tokens, "temperature": 0.2,
        "messages": [{"role": "user", "content": content}]
    }

    for attempt in range(MAX_API_RETRIES):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=API_TIMEOUT)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.warning(f"API Request Failed (Attempt {attempt + 1}/{MAX_API_RETRIES}): {e}. Retrying...")
            time.sleep(2 ** attempt)
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse API response: {e}. Response: {response.text}")
            return None
    
    st.error("An API error occurred. Please check the backend logs for details.")
    logger.error("API call ultimately failed after all retries.")
    return None

def get_text_from_pdf_ocr(file_content: bytes, filename_for_status: str) -> str:
    """Extracts full text from a PDF's content using vision model OCR."""
    images = pdf_to_images(file_content)
    if not images: return ""
    
    full_text = ""
    for i, page_image in enumerate(images):
        try:
            img_buffer = io.BytesIO()
            page_image.save(img_buffer, format='PNG')
            image_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            ocr_prompt = "Extract all text from this document image. Transcribe the text exactly as it appears."
            ocr_result = call_together_api(ocr_prompt, image_b64)
            if ocr_result:
                full_text += ocr_result + "\n\n--- PAGE BREAK ---\n\n"
        except Exception as e:
            logger.error(f"Error processing page {i+1} of {filename_for_status}: {e}")
            continue
    return full_text

def get_text_from_docx(file_content: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_content))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error processing DOCX file: {e}"); return ""

def get_text_from_excel(file_content: bytes) -> str:
    try:
        df_dict = pd.read_excel(io.BytesIO(file_content), sheet_name=None)
        return "\n\n".join([f"--- Content from Sheet: {name} ---\n{df.to_string(index=False)}" for name, df in df_dict.items()])
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}"); return ""

def get_text_from_csv(file_content: bytes) -> str:
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        return df.to_string(index=False)
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}"); return ""

@st.cache_data(show_spinner=False)
def extract_text_from_file(filename: str, file_content: bytes) -> str:
    """Master function to extract text from any supported file type."""
    file_ext = filename.lower().split('.')[-1]
    logger.info(f"Extracting text from '{filename}' (type: {file_ext}).")
    
    extractors = {
        'pdf': get_text_from_pdf_ocr,
        'xlsx': get_text_from_excel, 'xls': get_text_from_excel,
        'docx': get_text_from_docx,
        'csv': get_text_from_csv,
    }
    
    if file_ext in extractors:
        # Pass filename only to the ocr function
        if file_ext == 'pdf':
            return extractors[file_ext](file_content, filename)
        return extractors[file_ext](file_content)
    elif file_ext in ['txt', 'md']:
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error decoding text file {filename}: {e}"); return ""
    else:
        logger.warning(f"Unsupported file type encountered: {filename}"); return ""

def extract_document_metadata(text: str) -> Dict[str, Any]:
    """Extracts key metadata from document text."""
    prompt = f"Analyze the text. Extract 'work_order' (number, null if none), 'client' (name, null if none), 'heading', and a one-sentence 'summary'. Respond in JSON. Text: {text[:4000]}"
    response = call_together_api(prompt)
    if not response: return {"work_order": None, "client": None, "heading": "Unknown", "summary": "Metadata extraction failed."}
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.warning("Could not parse valid JSON metadata."); return {"work_order": None, "client": None, "heading": "Unknown", "summary": "Could not generate summary."}

def process_file_for_index(filename: str) -> Optional[Dict[str, Any]]:
    """Worker function to process a single file for the database index."""
    try:
        with open(os.path.join(DATABASE_DIR, filename), 'rb') as f: content = f.read()
        text = extract_text_from_file(filename, content)
        if text:
            text_cache_path = os.path.join(OCR_CACHE_DIR, filename + ".txt")
            with open(text_cache_path, 'w', encoding='utf-8') as cache_file: cache_file.write(text)
            metadata = extract_document_metadata(text)
            return {filename: metadata}
        else:
            logger.warning(f"Failed to extract text from '{filename}'. Skipping."); return None
    except Exception as e:
        logger.error(f"Unexpected error processing {filename}: {e}"); return None

@st.cache_resource
def automatic_database_build():
    """Automatically builds or updates the database index at startup."""
    for dir_path in [DATABASE_DIR, OCR_CACHE_DIR]:
        if not os.path.exists(dir_path): os.makedirs(dir_path); logger.info(f"Created directory: {dir_path}")

    index = {}
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, 'r') as f: index = json.load(f)
        except json.JSONDecodeError: logger.warning("Index file corrupted. Starting fresh."); index = {}

    db_files = {f for f in os.listdir(DATABASE_DIR) if f.lower().endswith(SUPPORTED_FILE_TYPES)}
    indexed_files = set(index.keys())
    files_to_process = list(db_files - indexed_files)

    if not files_to_process:
        logger.info("Database index is up to date."); return

    with st.spinner(f"Found {len(files_to_process)} new document(s). Indexing now..."):
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_filename = {executor.submit(process_file_for_index, filename): filename for filename in files_to_process}
            for future in as_completed(future_to_filename):
                result = future.result()
                if result: index.update(result)
        
        with open(INDEX_FILE, 'w') as f: json.dump(index, f, indent=4)
    st.toast(f"✅ Successfully added {len(files_to_process)} new document(s) to the database!")
    logger.info(f"Database update complete. Added {len(files_to_process)} files.")


# --- Chatbot Core Functions ---

def get_user_intent(user_prompt: str) -> str:
    """Uses the LLM to determine the user's intent."""
    intent_prompt = f"Classify the user's request as 'compare_documents', 'find_similar', or 'general_question'. Request: '{user_prompt}'"
    intent = call_together_api(intent_prompt, max_tokens=50)
    if intent:
        cleaned_intent = intent.strip().replace("'", "").replace('"', '').lower()
        if cleaned_intent in ['compare_documents', 'find_similar', 'general_question']: return cleaned_intent
    logger.warning(f"Could not determine intent for prompt. Defaulting to 'general_question'.")
    return 'general_question'

def find_similar_documents() -> str:
    """Finds documents in the database with the same work order as the uploaded file."""
    if not st.session_state.uploaded_file_data: return "Please upload a file first."
    if not os.path.exists(INDEX_FILE): return "Database index not found."

    with open(INDEX_FILE, 'r') as f: db_index = json.load(f)
    uploaded_wo = str(st.session_state.uploaded_file_data.get("work_order", "")).strip()
    if not uploaded_wo: return "The uploaded file does not have a Work Order number."

    similar_docs = [fname for fname, fmeta in db_index.items() if str(fmeta.get("work_order", "")).strip() == uploaded_wo]
    
    if not similar_docs: return f"No documents found in the database with Work Order: **{uploaded_wo}**."
    else:
        st.session_state.comparison_target = similar_docs
        doc_list = "\n".join([f"- `{doc}`" for doc in similar_docs])
        return f"I found {len(similar_docs)} document(s) with the same Work Order ({uploaded_wo}):\n{doc_list}\n\nDo you want me to compare the uploaded file with one of these?"

def perform_comparison(doc_b_name: str) -> str:
    """Performs a detailed comparison between two documents."""
    doc_a_name = st.session_state.uploaded_file_data["name"]
    doc_a_text = st.session_state.uploaded_file_data["text"]
    doc_b_text_path = os.path.join(OCR_CACHE_DIR, doc_b_name + ".txt")
    if not os.path.exists(doc_b_text_path): return f"Error: Could not find cached text for `{doc_b_name}`."
    
    with open(doc_b_text_path, 'r', encoding='utf-8') as f: doc_b_text = f.read()

    prompt = f"Perform a detailed comparison of Document A ({doc_a_name}) and Document B ({doc_b_name}). Create a markdown table contrasting all attributes. Document A Text: {doc_a_text} \n\n Document B Text: {doc_b_text}"
    with st.spinner(f"Generating comparison with `{doc_b_name}`..."):
        return call_together_api(prompt) or "Sorry, comparison failed."

def answer_uploaded_file_question(user_prompt: str) -> Optional[str]:
    """Answers a question using only the uploaded file's context."""
    context = st.session_state.uploaded_file_data.get("text", "")
    doc_name = st.session_state.uploaded_file_data.get("name", "the uploaded file")
    prompt = f"Answer the user's question using ONLY the context from `{doc_name}`. If the answer is not found, respond with the exact string 'ANSWER_NOT_FOUND'.\n\nContext: {context}\n\nQuestion: {user_prompt}"
    return call_together_api(prompt)

def answer_database_question(user_prompt: str) -> str:
    """Answers a question by finding relevant documents and using their content."""
    if not os.path.exists(INDEX_FILE): return "The document database has not been built."

    with open(INDEX_FILE, 'r') as f: index = json.load(f)
    summaries_context = "\n".join([f"File: {fname}\nSummary: {meta.get('summary', 'N/A')}" for fname, meta in index.items()])
    relevance_prompt = f"Based on the summaries, which files are relevant to the question: '{user_prompt}'? Respond with a JSON list of filenames. Summaries: {summaries_context}"
    
    with st.spinner("Scanning database for relevant documents..."):
        response = call_together_api(relevance_prompt, max_tokens=1024)
    
    relevant_doc_names = []
    if response:
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match: relevant_doc_names = json.loads(json_match.group(0))
        except json.JSONDecodeError: logger.error(f"Failed to decode JSON from relevance check: {response}")

    if not relevant_doc_names: return "I scanned the database but couldn't find any documents relevant to your question."

    with st.expander("Found relevant information in the following documents:", expanded=False):
        for name in relevant_doc_names: st.markdown(f"- `{name}`")

    with st.spinner("Synthesizing answer from relevant documents..."):
        relevant_docs_text = []
        for filename in relevant_doc_names:
            text_path = os.path.join(OCR_CACHE_DIR, filename + ".txt")
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f: relevant_docs_text.append(f"--- START: {filename} ---\n{f.read()}\n--- END: {filename} ---")
        
        context_text = "\n\n".join(relevant_docs_text)
        qa_prompt = f"Answer the user's question using ONLY the context from the relevant documents. Synthesize the answer and cite the source document(s).\n\nContext: {context_text}\n\nQuestion: {user_prompt}"
        return call_together_api(qa_prompt) or "Sorry, I could not find an answer in the relevant documents."

def handle_general_question(user_prompt: str) -> str:
    """Intelligently answers a question by first checking the uploaded file, then the database."""
    if st.session_state.uploaded_file_data.get("text"):
        with st.spinner(f"Searching in uploaded file: `{st.session_state.uploaded_file_data['name']}`..."):
            answer = answer_uploaded_file_question(user_prompt)
        if answer and "ANSWER_NOT_FOUND" not in answer: return answer
        st.info("Couldn't find an answer in the uploaded file. Expanding search to the entire database...")

    return answer_database_question(user_prompt)

# --- Main Streamlit UI ---
def main():
    """Main function to run the Streamlit application."""
    st.title("📄 Document Assistant")

    if not API_KEY:
        st.error("FATAL: TOGETHER_API_KEY environment variable not set. Application cannot start."); st.stop()

    automatic_database_build()

    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Upload a file for temporary analysis", 
            type=SUPPORTED_FILE_TYPES, 
            label_visibility="collapsed",
            key="file_uploader_widget"  # Added a unique key here
        )
        
        if uploaded_file:
            if st.session_state.get("uploaded_file_data", {}).get("name") != uploaded_file.name:
                with st.spinner(f"Analyzing `{uploaded_file.name}`..."):
                    content = uploaded_file.read()
                    text = extract_text_from_file(uploaded_file.name, content)
                    metadata = extract_document_metadata(text)
                    st.session_state.uploaded_file_data = {"name": uploaded_file.name, "text": text, **metadata}
                    st.toast(f"✅ Analyzed `{uploaded_file.name}`")
                    st.session_state.comparison_target = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking..."):
                intent = get_user_intent(prompt)
                
                if intent == 'compare_documents':
                    if not st.session_state.uploaded_file_data:
                        full_response = "Please upload a document to compare first."
                    else:
                        target_doc = next((doc for doc in (st.session_state.comparison_target or []) if doc in prompt), None)
                        if target_doc:
                            full_response = perform_comparison(target_doc)
                        else:
                            full_response = find_similar_documents()
                elif intent == 'find_similar':
                    full_response = find_similar_documents()
                else: # 'general_question'
                    full_response = handle_general_question(prompt)

            placeholder.markdown(full_response or "Sorry, I couldn't process that request.")
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
import streamlit as st
import requests
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import json
import shutil
from dotenv import load_dotenv
import time
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Assistant",
    page_icon="🧠",
    layout="wide"
)

# --- Removed custom styling to allow Streamlit's default dark/light theme ---


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Outputs logs to the console/terminal for production monitoring
    ]
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Constants & Initialization ---
DATABASE_DIR = "Uploads"
OCR_CACHE_DIR = "ocr_cache"
INDEX_FILE = "index.json"
API_KEY = os.getenv("TOGETHER_API_KEY")
SUPPORTED_FILE_TYPES = ('pdf', 'xlsx', 'xls', 'docx', 'txt', 'csv', 'md')
MAX_API_RETRIES = 3
API_TIMEOUT = 180 # seconds

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your documents today?"}]
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = {}
if "comparison_target" not in st.session_state:
    st.session_state.comparison_target = None


# --- Core API & File Processing Functions ---

@st.cache_data(show_spinner=False)
def pdf_to_images(file_content: bytes) -> List[Image.Image]:
    """Converts PDF file content into a list of PIL Images."""
    images = []
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        for page_num, page in enumerate(pdf_document):
            try:
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                buf = io.BytesIO(img_bytes)
                images.append(Image.open(buf))
            except Exception as e:
                logger.error(f"Failed to process page {page_num} of a PDF: {e}")
    except Exception as e:
        logger.error(f"Could not open PDF file for image conversion: {e}")
    return images

def call_together_api(prompt: str, image_base64: Optional[str] = None, max_tokens: int = 8192) -> Optional[str]:
    """Generic function to call Together AI API with retries and error handling."""
    if not API_KEY:
        logger.critical("TOGETHER_API_KEY is not configured.")
        return None

    endpoint = 'https://api.together.xyz/v1/chat/completions'
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    content = [{"type": "text", "text": prompt}]
    if image_base64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})

    payload = {
        "model": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "max_tokens": max_tokens, "temperature": 0.2,
        "messages": [{"role": "user", "content": content}]
    }

    for attempt in range(MAX_API_RETRIES):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=API_TIMEOUT)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.warning(f"API Request Failed (Attempt {attempt + 1}/{MAX_API_RETRIES}): {e}. Retrying...")
            time.sleep(2 ** attempt)
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse API response: {e}. Response: {response.text}")
            return None
    
    st.error("An API error occurred. Please check the backend logs for details.")
    logger.error("API call ultimately failed after all retries.")
    return None

def get_text_from_pdf_ocr(file_content: bytes, filename_for_status: str) -> str:
    """Extracts full text from a PDF's content using vision model OCR."""
    images = pdf_to_images(file_content)
    if not images: return ""
    
    full_text = ""
    for i, page_image in enumerate(images):
        try:
            img_buffer = io.BytesIO()
            page_image.save(img_buffer, format='PNG')
            image_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            ocr_prompt = "Extract all text from this document image. Transcribe the text exactly as it appears."
            ocr_result = call_together_api(ocr_prompt, image_b64)
            if ocr_result:
                full_text += ocr_result + "\n\n--- PAGE BREAK ---\n\n"
        except Exception as e:
            logger.error(f"Error processing page {i+1} of {filename_for_status}: {e}")
            continue
    return full_text

def get_text_from_docx(file_content: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_content))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error processing DOCX file: {e}"); return ""

def get_text_from_excel(file_content: bytes) -> str:
    try:
        df_dict = pd.read_excel(io.BytesIO(file_content), sheet_name=None)
        return "\n\n".join([f"--- Content from Sheet: {name} ---\n{df.to_string(index=False)}" for name, df in df_dict.items()])
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}"); return ""

def get_text_from_csv(file_content: bytes) -> str:
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        return df.to_string(index=False)
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}"); return ""

@st.cache_data(show_spinner=False)
def extract_text_from_file(filename: str, file_content: bytes) -> str:
    """Master function to extract text from any supported file type."""
    file_ext = filename.lower().split('.')[-1]
    logger.info(f"Extracting text from '{filename}' (type: {file_ext}).")
    
    extractors = {
        'pdf': get_text_from_pdf_ocr,
        'xlsx': get_text_from_excel, 'xls': get_text_from_excel,
        'docx': get_text_from_docx,
        'csv': get_text_from_csv,
    }
    
    if file_ext in extractors:
        # Pass filename only to the ocr function
        if file_ext == 'pdf':
            return extractors[file_ext](file_content, filename)
        return extractors[file_ext](file_content)
    elif file_ext in ['txt', 'md']:
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error decoding text file {filename}: {e}"); return ""
    else:
        logger.warning(f"Unsupported file type encountered: {filename}"); return ""

def extract_document_metadata(text: str) -> Dict[str, Any]:
    """Extracts key metadata from document text."""
    prompt = f"Analyze the text. Extract 'work_order' (number, null if none), 'client' (name, null if none), 'heading', and a one-sentence 'summary'. Respond in JSON. Text: {text[:4000]}"
    response = call_together_api(prompt)
    if not response: return {"work_order": None, "client": None, "heading": "Unknown", "summary": "Metadata extraction failed."}
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.warning("Could not parse valid JSON metadata."); return {"work_order": None, "client": None, "heading": "Unknown", "summary": "Could not generate summary."}

def process_file_for_index(filename: str) -> Optional[Dict[str, Any]]:
    """Worker function to process a single file for the database index."""
    try:
        with open(os.path.join(DATABASE_DIR, filename), 'rb') as f: content = f.read()
        text = extract_text_from_file(filename, content)
        if text:
            text_cache_path = os.path.join(OCR_CACHE_DIR, filename + ".txt")
            with open(text_cache_path, 'w', encoding='utf-8') as cache_file: cache_file.write(text)
            metadata = extract_document_metadata(text)
            return {filename: metadata}
        else:
            logger.warning(f"Failed to extract text from '{filename}'. Skipping."); return None
    except Exception as e:
        logger.error(f"Unexpected error processing {filename}: {e}"); return None

@st.cache_resource
def automatic_database_build():
    """Automatically builds or updates the database index at startup."""
    for dir_path in [DATABASE_DIR, OCR_CACHE_DIR]:
        if not os.path.exists(dir_path): os.makedirs(dir_path); logger.info(f"Created directory: {dir_path}")

    index = {}
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, 'r') as f: index = json.load(f)
        except json.JSONDecodeError: logger.warning("Index file corrupted. Starting fresh."); index = {}

    db_files = {f for f in os.listdir(DATABASE_DIR) if f.lower().endswith(SUPPORTED_FILE_TYPES)}
    indexed_files = set(index.keys())
    files_to_process = list(db_files - indexed_files)

    if not files_to_process:
        logger.info("Database index is up to date."); return

    with st.spinner(f"Found {len(files_to_process)} new document(s). Indexing now..."):
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_filename = {executor.submit(process_file_for_index, filename): filename for filename in files_to_process}
            for future in as_completed(future_to_filename):
                result = future.result()
                if result: index.update(result)
        
        with open(INDEX_FILE, 'w') as f: json.dump(index, f, indent=4)
    st.toast(f"✅ Successfully added {len(files_to_process)} new document(s) to the database!")
    logger.info(f"Database update complete. Added {len(files_to_process)} files.")


# --- Chatbot Core Functions ---

def get_user_intent(user_prompt: str) -> str:
    """Uses the LLM to determine the user's intent."""
    intent_prompt = f"Classify the user's request as 'compare_documents', 'find_similar', or 'general_question'. Request: '{user_prompt}'"
    intent = call_together_api(intent_prompt, max_tokens=50)
    if intent:
        cleaned_intent = intent.strip().replace("'", "").replace('"', '').lower()
        if cleaned_intent in ['compare_documents', 'find_similar', 'general_question']: return cleaned_intent
    logger.warning(f"Could not determine intent for prompt. Defaulting to 'general_question'.")
    return 'general_question'

def find_similar_documents() -> str:
    """Finds documents in the database with the same work order as the uploaded file."""
    if not st.session_state.uploaded_file_data: return "Please upload a file first."
    if not os.path.exists(INDEX_FILE): return "Database index not found."

    with open(INDEX_FILE, 'r') as f: db_index = json.load(f)
    uploaded_wo = str(st.session_state.uploaded_file_data.get("work_order", "")).strip()
    if not uploaded_wo: return "The uploaded file does not have a Work Order number."

    similar_docs = [fname for fname, fmeta in db_index.items() if str(fmeta.get("work_order", "")).strip() == uploaded_wo]
    
    if not similar_docs: return f"No documents found in the database with Work Order: **{uploaded_wo}**."
    else:
        st.session_state.comparison_target = similar_docs
        doc_list = "\n".join([f"- `{doc}`" for doc in similar_docs])
        return f"I found {len(similar_docs)} document(s) with the same Work Order ({uploaded_wo}):\n{doc_list}\n\nDo you want me to compare the uploaded file with one of these?"

def perform_comparison(doc_b_name: str) -> str:
    """Performs a detailed comparison between two documents."""
    doc_a_name = st.session_state.uploaded_file_data["name"]
    doc_a_text = st.session_state.uploaded_file_data["text"]
    doc_b_text_path = os.path.join(OCR_CACHE_DIR, doc_b_name + ".txt")
    if not os.path.exists(doc_b_text_path): return f"Error: Could not find cached text for `{doc_b_name}`."
    
    with open(doc_b_text_path, 'r', encoding='utf-8') as f: doc_b_text = f.read()

    prompt = f"Perform a detailed comparison of Document A ({doc_a_name}) and Document B ({doc_b_name}). Create a markdown table contrasting all attributes. Document A Text: {doc_a_text} \n\n Document B Text: {doc_b_text}"
    with st.spinner(f"Generating comparison with `{doc_b_name}`..."):
        return call_together_api(prompt) or "Sorry, comparison failed."

def answer_uploaded_file_question(user_prompt: str) -> Optional[str]:
    """Answers a question using only the uploaded file's context."""
    context = st.session_state.uploaded_file_data.get("text", "")
    doc_name = st.session_state.uploaded_file_data.get("name", "the uploaded file")
    prompt = f"Answer the user's question using ONLY the context from `{doc_name}`. If the answer is not found, respond with the exact string 'ANSWER_NOT_FOUND'.\n\nContext: {context}\n\nQuestion: {user_prompt}"
    return call_together_api(prompt)

def answer_database_question(user_prompt: str) -> str:
    """Answers a question by finding relevant documents and using their content."""
    if not os.path.exists(INDEX_FILE): return "The document database has not been built."

    with open(INDEX_FILE, 'r') as f: index = json.load(f)
    summaries_context = "\n".join([f"File: {fname}\nSummary: {meta.get('summary', 'N/A')}" for fname, meta in index.items()])
    relevance_prompt = f"Based on the summaries, which files are relevant to the question: '{user_prompt}'? Respond with a JSON list of filenames. Summaries: {summaries_context}"
    
    with st.spinner("Scanning database for relevant documents..."):
        response = call_together_api(relevance_prompt, max_tokens=1024)
    
    relevant_doc_names = []
    if response:
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match: relevant_doc_names = json.loads(json_match.group(0))
        except json.JSONDecodeError: logger.error(f"Failed to decode JSON from relevance check: {response}")

    if not relevant_doc_names: return "I scanned the database but couldn't find any documents relevant to your question."

    with st.expander("Found relevant information in the following documents:", expanded=False):
        for name in relevant_doc_names: st.markdown(f"- `{name}`")

    with st.spinner("Synthesizing answer from relevant documents..."):
        relevant_docs_text = []
        for filename in relevant_doc_names:
            text_path = os.path.join(OCR_CACHE_DIR, filename + ".txt")
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f: relevant_docs_text.append(f"--- START: {filename} ---\n{f.read()}\n--- END: {filename} ---")
        
        context_text = "\n\n".join(relevant_docs_text)
        qa_prompt = f"Answer the user's question using ONLY the context from the relevant documents. Synthesize the answer and cite the source document(s).\n\nContext: {context_text}\n\nQuestion: {user_prompt}"
        return call_together_api(qa_prompt) or "Sorry, I could not find an answer in the relevant documents."

def handle_general_question(user_prompt: str) -> str:
    """Intelligently answers a question by first checking the uploaded file, then the database."""
    if st.session_state.uploaded_file_data.get("text"):
        with st.spinner(f"Searching in uploaded file: `{st.session_state.uploaded_file_data['name']}`..."):
            answer = answer_uploaded_file_question(user_prompt)
        if answer and "ANSWER_NOT_FOUND" not in answer: return answer
        st.info("Couldn't find an answer in the uploaded file. Expanding search to the entire database...")

    return answer_database_question(user_prompt)

# --- Main Streamlit UI ---
def main():
    """Main function to run the Streamlit application."""
    st.title("📄 Document Assistant")

    if not API_KEY:
        st.error("FATAL: TOGETHER_API_KEY environment variable not set. Application cannot start."); st.stop()

    automatic_database_build()

    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Upload a file for temporary analysis", 
            type=SUPPORTED_FILE_TYPES, 
            label_visibility="collapsed",
            key="file_uploader_widget"  # Added a unique key here
        )
        
        if uploaded_file:
            if st.session_state.get("uploaded_file_data", {}).get("name") != uploaded_file.name:
                with st.spinner(f"Analyzing `{uploaded_file.name}`..."):
                    content = uploaded_file.read()
                    text = extract_text_from_file(uploaded_file.name, content)
                    metadata = extract_document_metadata(text)
                    st.session_state.uploaded_file_data = {"name": uploaded_file.name, "text": text, **metadata}
                    st.toast(f"✅ Analyzed `{uploaded_file.name}`")
                    st.session_state.comparison_target = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking..."):
                intent = get_user_intent(prompt)
                
                if intent == 'compare_documents':
                    if not st.session_state.uploaded_file_data:
                        full_response = "Please upload a document to compare first."
                    else:
                        target_doc = next((doc for doc in (st.session_state.comparison_target or []) if doc in prompt), None)
                        if target_doc:
                            full_response = perform_comparison(target_doc)
                        else:
                            full_response = find_similar_documents()
                elif intent == 'find_similar':
                    full_response = find_similar_documents()
                else: # 'general_question'
                    full_response = handle_general_question(prompt)

            placeholder.markdown(full_response or "Sorry, I couldn't process that request.")
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
