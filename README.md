\# Enterprise AI Knowledge Assistant



An AI-powered application that allows users to interact with PDF documents, retrieve relevant information, and generate intelligent answers using Generative AI.



\## Project Overview



The Enterprise AI Knowledge Assistant helps users extract information from PDF documents and ask questions in natural language.



It uses Retrieval-Augmented Generation (RAG) to find relevant document content and generate context-aware answers.



\## Features



\- Upload multiple PDF documents

\- Extract text from PDF files

\- Split documents into searchable text chunks

\- Generate semantic embeddings

\- Retrieve relevant information using similarity search

\- Generate answers using Google Gemini

\- Display source filenames and PDF page numbers

\- Maintain chat history

\- Clear chat functionality

\- Three answer modes:

&#x20; - \*\*PDF Only:\*\* Answers using uploaded PDF content.

&#x20; - \*\*General AI:\*\* Answers using general AI knowledge.

&#x20; - \*\*Hybrid:\*\* Combines PDF information with general knowledge.



\## Technologies Used



\- Python

\- Streamlit

\- Google Gemini API

\- Sentence Transformers

\- Scikit-learn

\- NumPy

\- PyPDF

\- Python-dotenv



\## Project Workflow



1\. Upload PDF documents.

2\. Extract text from each PDF page.

3\. Divide extracted text into smaller chunks.

4\. Generate embeddings using Sentence Transformers.

5\. Convert user questions into embeddings.

6\. Retrieve relevant chunks using cosine similarity.

7\. Send retrieved information to Gemini.

8\. Display the generated answer with source references.



\## Installation and Setup



\### 1. Clone the repository



```bash

git clone https://github.com/naveena-genai/enterprise-ai-knowledge-assistant.git

```



\### 2. Navigate to the project folder



```bash

cd enterprise-ai-knowledge-assistant

```



\### 3. Create a virtual environment



```bash

python -m venv venv

```



\### 4. Activate the virtual environment



Windows:



```bash

venv\\Scripts\\activate

```



\### 5. Install dependencies



```bash

pip install streamlit pypdf sentence-transformers scikit-learn numpy google-genai python-dotenv

```



\### 6. Configure the API key



Create a `.env` file in the project directory and add:



```env

GEMINI\_API\_KEY=your\_api\_key\_here

```



Replace the placeholder with your own Gemini API key. Never upload your API key to GitHub.



\### 7. Run the application



```bash

python -m streamlit run app.py

```



\## Future Improvements



\- Support for Word and Excel documents

\- OCR support for scanned PDFs

\- Improved retrieval accuracy

\- Conversation memory improvements

\- Downloadable AI-generated summaries

\- User authentication



\## Author



\*\*Naveena\*\*



B.Sc. Chemistry Graduate | Aspiring Generative AI Developer



\## License



This project is licensed under the MIT License.

