import streamlit as st
from ocr import extract_text_from_pdf
import pandas as pd
from io import BytesIO
import docx
import openpyxl
from embedder import process_text_to_faiss, query_faiss_index


def extract_text_from_docx(file):
    doc = docx.Document(file)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return "\n".join(fullText)


def main():
    st.title("📂 Upload your Files for SR&ED AI Tool")

    uploaded_files = st.file_uploader(
        "Upload PDFs, CSVs, Word docs, Excel files, or Audio files (mp3, wav)",
        type=["pdf", "csv", "docx", "xlsx", "mp3", "wav"],
        accept_multiple_files=True
    )

    all_extracted_text = ""

    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} file(s):")
        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name} ({uploaded_file.type}, {uploaded_file.size} bytes)")
            filename = uploaded_file.name.lower()

            if filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())

            elif filename.endswith(".pdf"):
                st.write("PDF uploaded — ready for OCR processing.")
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                extracted_text = extract_text_from_pdf("temp.pdf")
                st.text_area("Extracted Text", extracted_text, height=300)
                all_extracted_text += extracted_text + "\n\n"

            elif filename.endswith(".docx"):
                st.write("Word document uploaded — extracting text.")
                text = extract_text_from_docx(uploaded_file)
                st.text_area("Extracted Text", text, height=300)
                all_extracted_text += text + "\n\n"

            elif filename.endswith(".xlsx"):
                st.write("Excel file uploaded — previewing first sheet.")
                wb = openpyxl.load_workbook(uploaded_file, read_only=True)
                sheet = wb.active
                data = sheet.values
                columns = next(data)
                df = pd.DataFrame(data, columns=columns)
                st.dataframe(df.head())

            elif filename.endswith(".mp3") or filename.endswith(".wav"):
                st.write("Audio file uploaded — ready for transcription.")
                # Add your audio transcription logic here

            else:
                st.write("Unsupported file type.")

        # Process all extracted text into FAISS index if any text extracted
        if all_extracted_text.strip():
            st.write("Embedding all extracted text into FAISS index...")
            process_text_to_faiss(all_extracted_text, index_path="data/faiss_index")

    # Query input
    query = st.text_input("Ask a question:")
    if query:
        try:
            results = query_faiss_index(query)
            for i, (doc, score) in enumerate(results):
                st.markdown(f"**Result {i+1} (Score: {score:.2f})**")
                st.write(doc.page_content)
                st.markdown("---")
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
