import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser.document_parser import DocumentParser
from logger.logging import setup_logging

logger = setup_logging()

class NoneParser(DocumentParser):
    """空解析器"""
    
    def parse(self, data: bytes) -> str:
        """返回空字符串"""
        logger.info(f"空解析器，不解析数据，默认返回空字符串")
        return ""
    
    def supports(self, content_type: str) -> bool:
        """空解析器，默认支持所有类型，不检查内容类型"""
        return True


if __name__ == "__main__":
    parser = NoneParser()
    with open("../data/HR.pdf", "rb") as f:
        data = f.read()
    result = parser.parse(data)
    print(result[:200])