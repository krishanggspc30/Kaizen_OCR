import streamlit as st
import requests
import base64
from PIL import Image
import io
import os
import zipfile
import pdfplumber
import json
from dotenv import load_dotenv

# --- Updated LangChain imports for Q&A ---
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- OCR helper functions ---
def encode_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def extract_text_with_llama(image: Image.Image, api_key: str) -> tuple[str, str]:
    """Extract text from an image via Together AI's Llama Vision model."""
    base64_image = encode_image_to_base64(image)
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please extract ALL the text from this image. Do NOT include any commentary or explanations—just return the extracted text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.1
    }

    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        content = resp.json()
        text = content['choices'][0]['message']['content']
        return text, None
    except Exception as e:
        return None, str(e)


# --- Multi-format processing helpers ---

def process_image_file(file, api_key: str) -> str:
    image = Image.open(file)
    text, error = extract_text_with_llama(image, api_key)
    return text or ""


def process_pdf_file(file, api_key: str) -> str:
    pages_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pil_img = page.to_image(resolution=150).original
            page_text, _ = extract_text_with_llama(pil_img, api_key)
            pages_text.append(page_text or "")
    return "\n\n".join(pages_text)


def process_zip_file(file, api_key: str) -> str:
    aggregated = []
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            ext = name.lower().split('.')[-1]
            with z.open(name) as member:
                if ext in ("png", "jpg", "jpeg", "webp"):
                    aggregated.append(process_image_file(member, api_key))
                elif ext == "pdf":
                    aggregated.append(process_pdf_file(member, api_key))
    return "\n\n".join(aggregated)


def main():
    # Custom CSS for production-ready styling
    st.markdown("""
    <style>
    .main-header {
        display: flex;
        align-items: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .logo-section {
        margin-right: 2rem;
    }
    .logo-placeholder {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .title-section h1 {
        margin: 0;
        color: #1f2937;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .upload-section {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #cbd5e1;
        margin: 2rem 0;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .status-ready {
        background: #dcfce7;
        color: #166534;
    }
    .status-error {
        background: #fee2e2;
        color: #dc2626;
    }
    .qa-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    .answer-box {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with logo and title
    st.markdown("""
    <div class="main-header">
        <div class="logo-section">
            <div class="logo-placeholder">AI</div>
        </div>
        <div class="title-section">
            <h1>Document AI Assistant</h1>
            <p class="subtitle">Extract text from documents and get intelligent answers</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- API Key Configuration (Simplified) ---
    api_key = ""
    
    # Check for API key in environment variables (GitHub Secrets or .env)
    if "TOGETHER_API_KEY" in os.environ:
        api_key = os.getenv("TOGETHER_API_KEY")
        st.markdown('<div class="status-badge status-ready">🟢 API Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-error">🔴 API Key Required</div>', unsafe_allow_html=True)
        
        # Configuration expander for API key setup
        with st.expander("⚙️ Configure API Key", expanded=True):
            manual_key = st.text_input(
                "Enter Together AI API Key",
                type="password",
                placeholder="sk-...",
                help="Get your API key from Together AI"
            )
            if manual_key:
                if st.checkbox("Remember for this session"):
                    st.session_state.api_key = manual_key
                    st.success("✅ API key stored in session")
                    st.rerun()
                api_key = manual_key
            elif 'api_key' in st.session_state:
                api_key = st.session_state.api_key
                st.info("✅ Using stored API key")
                if st.button("Clear stored key"):
                    del st.session_state.api_key
                    st.rerun()

    # --- Document Upload Section ---
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📄 Upload Document")
        mode = st.selectbox(
            "Choose input method:",
            ["📁 Upload Files", "🌐 From URL", "📝 Paste Text"],
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("### Supported Formats")
        st.markdown("• Images (PNG, JPG, JPEG, WebP)")
        st.markdown("• PDF Documents")
        st.markdown("• ZIP Archives")
        st.markdown("• Plain Text")

    extracted_text = ""

    if mode == "📁 Upload Files":
        files = st.file_uploader(
            "Drag and drop files here or click to browse",
            type=["png", "jpg", "jpeg", "webp", "pdf", "zip"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if files and api_key:
            with st.spinner("🔄 Processing documents..."):
                parts = []
                for f in files:
                    ext = f.name.lower().split('.')[-1]
                    if ext in ("png", "jpg", "jpeg", "webp"):
                        parts.append(process_image_file(f, api_key))
                    elif ext == "pdf":
                        parts.append(process_pdf_file(f, api_key))
                    elif ext == "zip":
                        parts.append(process_zip_file(f, api_key))
                extracted_text = "\n\n---\n\n".join(parts)
        elif files:
            st.warning("⚠️ Please configure your API key to process files.")

    elif mode == "🌐 From URL":
        url = st.text_input("Enter document URL", placeholder="https://example.com/document.pdf")
        if st.button("🔄 Fetch Document", type="primary"):
            if not api_key:
                st.warning("⚠️ Please configure your API key first.")
            else:
                with st.spinner("🔄 Fetching document..."):
                    try:
                        res = requests.get(url)
                        if res.ok:
                            buf = io.BytesIO(res.content)
                            if "pdf" in res.headers.get("Content-Type","") or url.lower().endswith(".pdf"):
                                extracted_text = process_pdf_file(buf, api_key)
                            else:
                                img = Image.open(buf)
                                extracted_text, _ = extract_text_with_llama(img, api_key)
                                extracted_text = extracted_text or ""
                        else:
                            st.error("❌ Failed to fetch document. Please check the URL.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    else:  # Paste text
        extracted_text = st.text_area(
            "Paste your text here", 
            height=200,
            placeholder="Paste or type your text here..."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Q&A Section ---
    if extracted_text:
        # Success indicator
        st.success("✅ Document processed successfully")
        
        # Document stats in a compact format
        word_count = len(extracted_text.split())
        char_count = len(extracted_text)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", f"{word_count:,}")
        with col2:
            st.metric("Characters", f"{char_count:,}")
        with col3:
            st.metric("Status", "Ready")

        # Q&A Interface
        st.markdown('<div class="qa-section">', unsafe_allow_html=True)
        st.markdown("### 🤖 Ask Questions About Your Document")
        
        if not api_key:
            st.warning("⚠️ API key required for Q&A functionality")
        else:
            # Prepare documents for Q&A
            splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(extracted_text)
            docs = [Document(page_content=chunk) for chunk in chunks]

            # Initialize LLM and QA chain
            llm = ChatOpenAI(
                model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                api_key=api_key,
                base_url="https://api.together.xyz/v1",
                temperature=0.1,
                max_tokens=4000
            )
            qa_chain = load_qa_chain(llm, chain_type="stuff")

            # Question input
            question = st.text_input(
                "What would you like to know about this document?",
                placeholder="e.g., What are the main points discussed?",
                key="qa_question"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
            
            if (question and ask_button) or (question and st.session_state.get('auto_submit', False)):
                with st.spinner("🤖 Analyzing document..."):
                    try:
                        answer = qa_chain.run(input_documents=docs, question=question)
                        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                        st.markdown("**Answer:**")
                        st.write(answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Optional: Show document preview toggle (hidden by default)
        with st.expander("📄 View Extracted Text", expanded=False):
            st.text_area("Document Content:", value=extracted_text, height=300, disabled=True)

    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6b7280;">
            <h3>📄 No Document Loaded</h3>
            <p>Upload a document, enter a URL, or paste text to get started</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
