import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser.document_parser import DocumentParser
from logger.logging import setup_logging

logger = setup_logging()

class TextParser(DocumentParser):
    """纯文本文档解析器"""
    
    def parse(self, data: bytes) -> str:
        """解析文本文档"""
        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    return data.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            # 如果所有编码都失败，使用错误处理
            return data.decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.error(f"文本解析失败: {e}")
            return ""
    
    def supports(self, content_type: str) -> bool:
        """检查是否支持文本"""
        return content_type.lower() in ['text/plain', 'text/html', 'text/markdown', 'txt']


if __name__ == "__main__":
    parser = TextParser()
    with open("../data/text.txt", "rb") as f:
        data = f.read()
    result = parser.parse(data)
    print(result[:200])