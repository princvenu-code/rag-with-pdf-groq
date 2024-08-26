import os
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader, CSVLoader, UnstructuredExcelLoader, Docx2txtLoader
from chromadb.config import Settings

DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    # ".pdf": PDFMinerLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}


ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSISTENT_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

CHROMA_SETTINGS = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    )



