import streamlit as st
import requests
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Constants ---
st.set_page_config(
    page_title="Intelligent PDF Comparator 🧠",
    page_icon="🤖",
    layout="wide"
)

DATABASE_DIR = "uploads"
INDEX_FILE = "index.json"

# --- Core API & PDF Processing Functions ---

def pdf_to_images(file_content):
    """Converts PDF file content into a list of PIL Images."""
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        images = []
        for page in pdf_document:
            pix = page.get_pixmap()
            # CORRECTED: Use .tobytes() for modern PyMuPDF versions
            img_bytes = pix.tobytes("png")
            buf = io.BytesIO(img_bytes)
            images.append(Image.open(buf))
        return images
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

def call_together_api(api_key, prompt, image_base64=None):
    """Generic function to call Together AI API for either text or vision tasks."""
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    content = [{"type": "text", "text": prompt}]
    # Default to the vision model if an image is provided
    model = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo" # Default to text model
    
    if image_base64:
        model = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo" # Switch to vision model
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})

    payload = {
        "model": model,
        "max_tokens": 4096, 
        "temperature": 0.1,
        "messages": [{"role": "user", "content": content}]
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response content: {e.response.text}")
    except (KeyError, IndexError) as e:
        st.error(f"Failed to parse API response: {e}")
    return None

def get_full_text(file_content, api_key):
    """Extracts full text from a PDF's content using OCR."""
    images = pdf_to_images(file_content)
    if not images: return ""
    
    full_text = ""
    st.write(f"Analyzing {len(images)} page(s) from the PDF...")
    progress_bar = st.progress(0)
    for i, page_image in enumerate(images):
        try:
            img_buffer = io.BytesIO()
            page_image.save(img_buffer, format='PNG')
            image_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            ocr_prompt = "Extract all text from this document image. Transcribe the text exactly as it appears, including tables and all formatting details."
            ocr_result = call_together_api(api_key, ocr_prompt, image_b64)
            if ocr_result:
                full_text += ocr_result + "\n--- PAGE BREAK ---\n"
            progress_bar.progress((i + 1) / len(images))
        except Exception as e:
            st.warning(f"Error processing page {i+1}: {e}")
            continue
    return full_text

def get_category_summary(text, api_key):
    """Gets a structured summary and categories for a given text."""
    prompt = f"""
    Analyze the following document text and provide a short, one-sentence summary of its main purpose and key subject matter. Also, list up to 5 keywords or categories that best describe its content.
    
    Format the output as a single, clean JSON object with keys 'summary' and 'categories'.
    Example: {{"summary": "A technical specification for GSRTC bus seats, work order 1650.", "categories": ["GSRTC", "bus seats", "technical specification", "work order", "configuration"]}}

    Document Text:
    ---
    {text[:8000]}
    ---
    """
    response = call_together_api(api_key, prompt)
    try:
        # Clean the response to ensure it's valid JSON
        cleaned_response = response[response.find('{'):response.rfind('}')+1]
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, TypeError, AttributeError):
        st.warning("Could not generate a valid JSON summary. Using raw response.")
        return {"summary": response or "Could not generate summary.", "categories": []}

# --- Database Indexing ---

def build_database_index(api_key):
    """Builds or updates the JSON index for the PDF database."""
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
        st.info(f"Created database directory at `{DATABASE_DIR}`. Please add your PDFs there.")
        return

    st.info("Building database index... This may take a while for new files.")
    index = {}
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r') as f:
            try:
                index = json.load(f)
            except json.JSONDecodeError:
                index = {}

    db_files = [f for f in os.listdir(DATABASE_DIR) if f.endswith(".pdf")]
    
    with st.status("Processing database files...") as status:
        for i, filename in enumerate(db_files):
            if filename not in index:
                status.update(label=f"Processing new file: `{filename}` ({i+1}/{len(db_files)})")
                with open(os.path.join(DATABASE_DIR, filename), 'rb') as f:
                    content = f.read()
                
                text = get_full_text(content, api_key)
                if text:
                    summary_data = get_category_summary(text, api_key)
                    index[filename] = {"text": text, **summary_data}
            
    with open(INDEX_FILE, 'w') as f:
        json.dump(index, f, indent=4)
    st.success("✅ Database index is up to date!")

# --- Streamlit UI ---

st.title("🧠 Intelligent PDF Comparator")
st.markdown("Upload a PDF to find the most similar document in the database and get a detailed comparison.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key = os.getenv("TOGETHER_API_KEY")

    if api_key:
        st.success("✅ API key loaded.")
    else:
        st.error("❗️ API key not found in `.env` file.")

    if st.button("Build/Refresh Database Index"):
        if api_key:
            build_database_index(api_key)
        else:
            st.warning("Please set your API key first.")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your PDF to compare", type="pdf")
    compare_button = st.button("Find Match & Compare", type="primary")

# --- Main Logic ---
if compare_button and uploaded_file:
    if not api_key:
        st.warning("Cannot proceed without an API key.")
    elif not os.path.exists(INDEX_FILE):
        st.warning("Database index not found. Please build it first using the sidebar button.")
    else:
        with open(INDEX_FILE, 'r') as f:
            db_index = json.load(f)
        
        if not db_index:
            st.error(f"The database is empty. Please add PDFs to the `{DATABASE_DIR}` folder and build the index.")
        else:
            st.markdown("---")
            
            with st.spinner("Step 1/4: Analyzing your uploaded PDF..."):
                uploaded_content = uploaded_file.read()
                uploaded_text = get_full_text(uploaded_content, api_key)
                uploaded_summary_data = get_category_summary(uploaded_text, api_key)
                st.success("✅ Uploaded PDF analyzed.")
            
            with st.spinner("Step 2/4: Finding the best match in the database..."):
                db_summaries = "\n".join([f"- {fname}: {data.get('summary', 'N/A')}" for fname, data in db_index.items()])
                
                match_prompt = f"""
                You are a document matching expert. Below is a summary for a new document. Your task is to identify the single best match from the list of documents in the database.
                Respond *only* with the exact filename of the best match and nothing else.

                New Document Summary:
                {uploaded_summary_data.get('summary')}

                Database Documents:
                {db_summaries}
                """
                
                best_match_filename = call_together_api(api_key, match_prompt).strip().replace("`", "")
                
                if best_match_filename not in db_index:
                    st.error(f"AI returned an invalid filename: `{best_match_filename}`. Could not find a match.")
                else:
                    st.success(f"✅ Best match found: `{best_match_filename}`")
                    
                    with st.spinner("Step 3/4: Preparing final comparison..."):
                        match_text = db_index[best_match_filename]['text']
                        
                        comparison_prompt = f"""You are an expert document analyst. Perform a detailed contextual comparison of the two documents below.

--- Document A: {uploaded_file.name} ---
{uploaded_text[:8000]}
--- End of Document A ---

--- Document B: {best_match_filename} ---
{match_text[:8000]}
--- End of Document B ---

Now, provide a detailed comparison.
1.  **High-Level Summary:** Briefly state the key similarities and differences.
2.  **Structured Comparison:** Create a markdown table that precisely contrasts specific attributes found in the text.
3.  **Conclusion:** Summarize what distinguishes each document's project or configuration.

Format the response using markdown.
"""
                    with st.spinner("Step 4/4: Generating final comparison... this may take a moment."):
                        final_comparison = call_together_api(api_key, comparison_prompt)
                    
                    st.balloons()
                    st.header("Comparison Result")
                    st.markdown(f"Comparing **{uploaded_file.name}** with **{best_match_filename}**:")
                    st.markdown(final_comparison or "Could not generate comparison.")
