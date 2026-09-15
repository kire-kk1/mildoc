from io import BytesIO
from PyPDF2 import PdfReader

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser.document_parser import DocumentParser
from logger.logging import setup_logging

logger = setup_logging()

class PDFParser(DocumentParser):
    """PDF文档解析器"""
    
    def parse(self, data: bytes) -> str:
        """解析PDF文档"""
        try:
            reader = PdfReader(BytesIO(data))
            text_content = ""
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"PDF解析失败: {e}")
            return ""
    
    def supports(self, content_type: str) -> bool:
        """检查是否支持PDF"""
        return content_type.lower() in ['application/pdf', 'pdf']


if __name__ == "__main__":
    parser = PDFParser()

    file_path = ["../data/pdf1.pdf", "../data/pdf2.pdf", "../data/pdf3.pdf", "../data/pdf4.pdf"]
    for path in file_path:
        with open(path, "rb") as f:
            data = f.read()
            result = parser.parse(data)
            print("\n\n")
            print(f"#################   文件路径: {path}    #################")
            print(result)

