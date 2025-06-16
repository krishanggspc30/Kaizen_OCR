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

# --- LangChain imports for Q&A ---
from langchain_community.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="OCR with Llama Vision + Q&A",
    page_icon="📄🤖",
    layout="wide"
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
    st.title("📄 OCR with Llama Vision + Q&A")
    st.markdown("Upload documents (images/PDFs/ZIPs), fetch from URL, or paste text—then ask questions.")

    # --- Sidebar: API Key Configuration ---
    api_key = ""
    with st.sidebar:
        st.header("🔧 Configuration")
        if "TOGETHER_API_KEY" in os.environ:
            api_key = os.getenv("TOGETHER_API_KEY")
            st.markdown("**API Key loaded from environment variable.**")
        else:
            manual_key = st.text_input(
                "Enter Together AI API Key",
                type="password",
                placeholder="sk-...",
                help="Used for OCR and Q&A requests."
            )
            if manual_key:
                if st.checkbox("Remember key for this session"):
                    st.session_state.api_key = manual_key
                    st.success("🔒 API key stored in session.")
                api_key = manual_key
            elif 'api_key' in st.session_state:
                api_key = st.session_state.api_key
                st.info("Using stored API key from session.")

        if 'api_key' in st.session_state:
            if st.button("Clear stored API key"):
                del st.session_state.api_key
                st.experimental_rerun()

        st.markdown("---")
        st.markdown("### How to set API Key")
        st.markdown(
            """
1. Create a `.env` file:
```
TOGETHER_API_KEY=your_key_here
```
2. Or set env var:
```bash
export TOGETHER_API_KEY=your_key_here
```
3. Or enter manually above.
            """
        )

    st.markdown("---")
    st.markdown(f"**Status:** {'🟢 Ready' if api_key else '🔴 API Key Required'}")

    # --- Main Input Section ---
    st.header("📥 Input Document / Text")
    mode = st.selectbox(
        "Choose input type:",
        ["Upload file(s)", "Enter URL", "Paste text"]
    )

    extracted_text = ""

    if mode == "Upload file(s)":
        files = st.file_uploader(
            "Select images, PDFs, or ZIP",
            type=["png", "jpg", "jpeg", "webp", "pdf", "zip"],
            accept_multiple_files=True
        )
        if files and api_key:
            with st.spinner("Processing..."):
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

    elif mode == "Enter URL":
        url = st.text_input("Image or PDF URL")
        if st.button("Fetch"):
            if not api_key:
                st.warning("⚠️ Please configure your API key first.")
            else:
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
                    st.error("Failed to fetch URL. Please check the link.")

    else:  # Paste text
        pasted = st.text_area("Paste your text here")
        if pasted:
            extracted_text = pasted

    # --- Display & Q&A Section ---
    if extracted_text:
        st.success("✅ Text ready for Q&A")
        st.text_area("Extracted / Input Text:", value=extracted_text, height=300)

        # Stats
        word_count = len(extracted_text.split())
        char_count = len(extracted_text)
        st.markdown(f"**Word Count:** {word_count} &nbsp;&nbsp; **Character Count:** {char_count}")

        # Q&A
        st.markdown("---")
        st.header("❓ Ask a Question")
        st.markdown("Enter a question about the text above.")

        # Prepare documents
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(extracted_text)
        docs = [Document(page_content=chunk) for chunk in chunks]

        # Initialize LLM and QA chain
        llm = ChatOpenAI(
            model_name="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            openai_api_key=api_key,
            openai_api_base="https://api.together.xyz/v1",
            temperature=0.1,
            max_tokens=4000
        )
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        question = st.text_input(
            "Your question:",
            placeholder="e.g., What is the main instruction given?",
            key="qa_question"
        )
        if question:
            with st.spinner("Generating answer..."):
                answer = qa_chain.run(input_documents=docs, question=question)
            st.markdown("**Answer:**")
            st.write(answer)

    else:
        st.info("👆 Provide input above to extract text.")


if __name__ == "__main__":
    main()
