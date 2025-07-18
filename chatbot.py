import streamlit as st
import requests
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import json
from dotenv import load_dotenv

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Document Assistant �",
    page_icon="🤖",
    layout="wide"
)

# --- Load Environment Variables ---
load_dotenv()

# --- Constants & Initialization ---
DATABASE_DIR = "uploads"
INDEX_FILE = "index.json"
API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize session state for chat and uploaded file data
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_text" not in st.session_state:
    st.session_state.uploaded_file_text = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# --- Core API & PDF Processing Functions ---

def pdf_to_images(file_content):
    """Converts PDF file content into a list of PIL Images."""
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        images = []
        for page in pdf_document:
            pix = page.get_pixmap(dpi=150) # Increased DPI for better OCR
            img_bytes = pix.tobytes("png")
            buf = io.BytesIO(img_bytes)
            images.append(Image.open(buf))
        return images
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

def call_together_api(prompt, image_base64=None):
    """Generic function to call Together AI API for either text or vision tasks."""
    if not API_KEY:
        st.error("API key is not configured. Please check your .env file.")
        return None

    endpoint = 'https://api.together.xyz/v1/chat/completions'
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    content = [{"type": "text", "text": prompt}]
    model = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo" # Default to a strong text model
    
    if image_base64:
        model = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})

    payload = {
        "model": model,
        "max_tokens": 4096, 
        "temperature": 0.2, # Slightly higher temp for more creative chat
        "messages": [{"role": "user", "content": content}]
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response content: {e.response.text}")
    except (KeyError, IndexError) as e:
        st.error(f"Failed to parse API response: {e}")
    return None

def get_full_text_from_pdf(file_content):
    """Extracts full text from a PDF's content using vision model OCR."""
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
            ocr_result = call_together_api(ocr_prompt, image_b64)
            if ocr_result:
                full_text += ocr_result + "\n\n--- PAGE BREAK ---\n\n"
            progress_bar.progress((i + 1) / len(images))
        except Exception as e:
            st.warning(f"Error processing page {i+1}: {e}")
            continue
    return full_text

def get_category_summary(text):
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
    response = call_together_api(prompt)
    try:
        cleaned_response = response[response.find('{'):response.rfind('}')+1]
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, TypeError, AttributeError):
        st.warning("Could not generate a valid JSON summary. Using raw response.")
        return {"summary": response or "Could not generate summary.", "categories": []}

def build_database_index():
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
                
                text = get_full_text_from_pdf(content)
                if text:
                    summary_data = get_category_summary(text)
                    index[filename] = {"text": text, **summary_data}
            
    with open(INDEX_FILE, 'w') as f:
        json.dump(index, f, indent=4)
    st.success("✅ Database index is up to date!")

# --- Streamlit UI ---

st.title("Kaizen Engineering")

# --- Sidebar for Upload and Indexing ---
with st.sidebar:
    st.header("Configuration")
    if API_KEY:
        st.success("✅ API key loaded.")
    else:
        st.error("❗️ API key not found in `.env` file.")

    if st.button("Build/Refresh Database Index"):
        if API_KEY:
            build_database_index()
        else:
            st.warning("Please set your API key first.")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload a PDF for analysis", type="pdf")
    
    if uploaded_file and API_KEY:
        # Process uploaded file immediately and store its text in session state
        if st.session_state.uploaded_file_name != uploaded_file.name:
            with st.spinner("Analyzing your uploaded PDF..."):
                content = uploaded_file.read()
                st.session_state.uploaded_file_text = get_full_text_from_pdf(content)
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success(f"✅ Analyzed `{uploaded_file.name}`.")


# --- Main Application Tabs ---
tab1, tab2 = st.tabs(["💬 Chatbot Q&A", "↔️ PDF Comparator"])

# --- Chatbot Tab ---
with tab1:
    st.header("Conversational Q&A")
    st.markdown("Ask questions about your documents.")

    chat_context_option = st.selectbox(
        "Choose the context for your questions:",
        ("Uploaded PDF", "Entire PDF Database"),
        key="chat_context"
    )

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            context_text = ""
            # Gather context based on user's selection
            if chat_context_option == "Uploaded PDF":
                if st.session_state.uploaded_file_text:
                    context_text = st.session_state.uploaded_file_text
                    context_source = f"the document `{st.session_state.uploaded_file_name}`"
                else:
                    full_response = "Please upload a PDF first to ask questions about it."
            
            elif chat_context_option == "Entire PDF Database":
                if os.path.exists(INDEX_FILE):
                    with open(INDEX_FILE, 'r') as f:
                        db_index = json.load(f)
                    if db_index:
                        context_text = "\n\n".join([f"--- START OF DOCUMENT: {fname} ---\n{data['text']}\n--- END OF DOCUMENT: {fname} ---" for fname, data in db_index.items()])
                        context_source = "the entire document database"
                    else:
                        full_response = "The database is empty. Please add PDFs and build the index."
                else:
                    full_response = "Database index not found. Please build it first."

            if context_text:
                with st.spinner(f"Searching for answers in {context_source}..."):
                    qa_prompt = f"""You are an expert assistant. Using ONLY the context provided below, answer the user's question.
Your answer must be grounded in the text. If the answer is not found in the context, clearly state that.

--- CONTEXT ---
{context_text[:20000]} 
--- END OF CONTEXT ---

USER'S QUESTION: {prompt}
"""
                    full_response = call_together_api(qa_prompt)

            message_placeholder.markdown(full_response or "Sorry, I couldn't get a response from the API.")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# --- Comparator Tab ---
with tab2:
    st.header("Intelligent PDF Comparator")
    st.markdown("Find the most similar document in the database and get a detailed comparison.")
    
    if st.button("Find Match & Compare", type="primary"):
        if not st.session_state.uploaded_file_text:
            st.warning("Please upload a PDF file from the sidebar first.")
        elif not os.path.exists(INDEX_FILE):
            st.warning("Database index not found. Please build it first using the sidebar button.")
        else:
            with open(INDEX_FILE, 'r') as f:
                db_index = json.load(f)
            
            if not db_index:
                st.error(f"The database is empty. Please add PDFs to the `{DATABASE_DIR}` folder and build the index.")
            else:
                with st.spinner("Step 1/3: Analyzing uploaded PDF summary..."):
                    uploaded_summary_data = get_category_summary(st.session_state.uploaded_file_text)
                    st.success("✅ Uploaded PDF summary analyzed.")
                
                with st.spinner("Step 2/3: Finding the best match in the database..."):
                    db_summaries = "\n".join([f"- {fname}: {data.get('summary', 'N/A')}" for fname, data in db_index.items()])
                    
                    match_prompt = f"""
                    You are a document matching expert. Below is a summary for a new document. Your task is to identify the single best match from the list of documents in the database.
                    Respond *only* with the exact filename of the best match and nothing else.

                    New Document Summary:
                    {uploaded_summary_data.get('summary')}

                    Database Documents:
                    {db_summaries}
                    """
                    
                    best_match_filename = call_together_api(match_prompt).strip().replace("`", "")
                    
                    if best_match_filename not in db_index:
                        st.error(f"AI returned an invalid filename: `{best_match_filename}`. Could not find a match.")
                    else:
                        st.success(f"✅ Best match found: `{best_match_filename}`")
                        
                        with st.spinner("Step 3/3: Generating final comparison... this may take a moment."):
                            match_text = db_index[best_match_filename]['text']
                            
                            comparison_prompt = f"""You are an expert document analyst. Perform a detailed contextual comparison of the two documents below.

--- Document A: {st.session_state.uploaded_file_name} ---
{st.session_state.uploaded_file_text[:8000]}
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
                            final_comparison = call_together_api(comparison_prompt)
                        
                        st.balloons()
                        st.header("Comparison Result")
                        st.markdown(f"Comparing **{st.session_state.uploaded_file_name}** with **{best_match_filename}**:")
                        st.markdown(final_comparison or "Could not generate comparison.")
