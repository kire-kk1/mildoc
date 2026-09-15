from minio import Minio
from dotenv import load_dotenv
import os, sys
import hashlib
import argparse
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.document_parser import DocumentParser
from parser.office_parser import OfficeParser
from parser.pdf_parser import PDFParser
from parser.markdown_parser import MarkdownParser
from parser.text_parser import TextParser
from parser.none_parser import NoneParser
from logger.logging import setup_logging

load_dotenv()
logger = setup_logging()

class SimpleObjectParser:
    """简单对象解析器"""
    
    def __init__(self,minio_client: Minio, chunk_size: int = 2048, overlap_size: int = 128):
        """
        初始化解析器
        
        Args:
            chunk_size (int): 文本片段最大长度，默认2048
            overlap_size (int): 重叠区域大小，默认128
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
        # 初始化Minio客户端
        self.minio_client = minio_client
        
        # 注册解析器（按优先级排序）
        self.parsers = [
            PDFParser(),             # PyPDF2作为PDF解析器
            OfficeParser(),          # markitdown解析Office文档（Word、Excel、PowerPoint）
            #MinerUParser(),          # MinerU解析器，专门处理PDF，暂不开启
            MarkdownParser(),        # Markdown解析器，
            TextParser(),            # 纯文本解析器
            NoneParser(),            # 空解析器, 作为最后的备选
        ]
    
    def add_parser(self, parser: DocumentParser):
        """
        添加新的解析器
        
        Args:
            parser (DocumentParser): 文档解析器实例
        """
        self.parsers.append(parser)
    
    def _get_parser(self, content_type: str) -> Optional[DocumentParser]:
        """
        根据内容类型获取合适的解析器
        
        Args:
            content_type (str): 内容类型
            
        Returns:
            Optional[DocumentParser]: 解析器实例，如果没有找到则返回None
        """
        for parser in self.parsers:
            if parser.supports(content_type):
                return parser
        return None
    
    def _extract_doc_name(self, object_path: str) -> str:
        """
        从对象路径中提取文档名称
        
        Args:
            object_path (str): 对象路径
            
        Returns:
            str: 文档名称
        """
        return os.path.basename(object_path)
    
    def _extract_doc_type(self, content_type: str) -> str:
        """
        从content-type中提取文档类型
        
        Args:
            content_type (str): 内容类型
            
        Returns:
            str: 文档类型
        """
        if not content_type:
            return "unknown"
        
        # 提取主要类型
        main_type = content_type.split('/')[0].lower()
        sub_type = content_type.split('/')[-1].lower()
        
        # 映射常见类型
        type_mapping = {
            'application/pdf': 'pdf',
            'text/plain': 'txt',
            'text/html': 'html',
            'text/markdown': 'md',
            'text/x-markdown': 'md',
            'application/markdown': 'md',
            
            # Word文档
            'application/msword': 'doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            
            # Excel文档
            'application/vnd.ms-excel': 'xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            
            # PowerPoint文档
            'application/vnd.ms-powerpoint': 'ppt',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
        }
        
        return type_mapping.get(content_type.lower(), sub_type)
    
    def _calculate_md5(self, data: bytes) -> str:
        """
        计算数据的MD5值
        
        Args:
            data (bytes): 二进制数据
            
        Returns:
            str: MD5值
        """
        return hashlib.md5(data).hexdigest()
    
    def _split_text_by_langchain(self, text: str) -> List[str]:
        """
        使用LangChain分割文本
        """

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap_size)
        return text_splitter.split_text(text)
    
    def parse_object(self, bucket_name: str, object_name: str) -> Dict[str, Any]:
        """
        解析简单对象
        
        Args:
            bucket_name (str): 存储桶名称
            object_path (str): 对象路径
            
        Returns:
            Dict[str, Any]: 解析结果
        """
        try:
            # 先获取对象信息，检查文件大小
            logger.info(f"正在检查对象信息: {bucket_name}/{object_name}")
            try:
                stat = self.minio_client.stat_object(bucket_name, object_name)
                file_size = stat.size
                max_size = 512 * 1024 * 1024  # 512MB
                
                logger.info(f"对象大小: {file_size} 字节 ({file_size / 1024 / 1024:.2f} MB)")
                
                if file_size > max_size:
                    logger.info(f"文件过大 ({file_size / 1024 / 1024:.2f} MB > 512 MB)，跳过解析")
                    return None
            except Exception as e:
                logger.info(f"获取对象信息失败: {e}")
                # 如果无法获取文件信息，继续尝试解析
            
            # 从Minio获取对象
            logger.info(f"正在获取对象内容: {bucket_name}/{object_name}")
            response = self.minio_client.get_object(bucket_name, object_name)
            
            # 获取对象数据和元数据
            data = response.data
            headers = response.headers
            
            logger.info(f"对象大小: {len(data)} 字节")
            logger.info(f"Content-Type: {headers.get('Content-Type', 'unknown')}")
            
            # 提取基本信息
            doc_name = self._extract_doc_name(object_name)
            doc_path_name = object_name  # 不再包含bucket_name前缀
            content_type = headers.get('Content-Type', '')
            doc_type = self._extract_doc_type(content_type)
            doc_md5 = headers.get('ETag', '').strip('"')
            if len(doc_md5) != 32:  # 如果ETag不是32位，则重新计算MD5, 多部分上传：ETag = {复合MD5}-{部分数量}（超过32字符）
                doc_md5 = self._calculate_md5(data)
            #doc_md5 = self._calculate_md5(data)
            doc_length = int(headers.get('Content-Length', len(data)))
            
            # 选择合适的解析器
            parser = self._get_parser(content_type)
            if not parser:
                logger.info(f"警告: 未找到适合 {content_type} 的解析器，使用空解析器兜底")
                parser = NoneParser()
            
            # 解析文档内容
            logger.info(f"使用解析器: {parser.__class__.__name__}")
            text_content = parser.parse(data)
            
            if not text_content:
                logger.info("警告: 未提取到文本内容")
                contents = []
            else:
                logger.info(f"提取到文本: {len(text_content)} 字符")
                # 分割文本为片段
                contents = self._split_text_by_langchain(text_content)

                logger.info(f"分割为 {len(contents)} 个片段")
            
            return {
                "doc_name": doc_name,
                "doc_path_name": doc_path_name,
                "doc_type": doc_type,
                "doc_md5": doc_md5,
                "doc_length": doc_length,
                "contents": contents
            }
            
        except Exception as e:
            logger.info(f"解析对象失败: {e}")
            return None
        finally:
            # 确保响应被正确关闭
            if 'response' in locals():
                response.close()
                response.release_conn()
    
    def get_parser_info(self) -> List[str]:
        """
        获取已注册的解析器信息
        
        Returns:
            List[str]: 解析器类名列表
        """
        return [parser.__class__.__name__ for parser in self.parsers]


# 使用示例和测试
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='简单对象解析器 - 支持多种文档格式解析')
    
    # 基本参数
    parser.add_argument('--bucket', type=str, default='test', help='Minio桶名称 (默认: test)')
    parser.add_argument('--file', type=str, action='append', help='要解析的文件路径 (可指定多个文件)')
    parser.add_argument('--test', action='store_true', help='运行默认测试案例')
    
    # 解析器配置
    parser.add_argument('--chunk-size', type=int, default=2048, help='文本片段最大长度 (默认: 2048)')
    parser.add_argument('--overlap-size', type=int, default=128, help='重叠区域大小 (默认: 128)')
    
    # 输出选项
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    print("=== 简单对象解析器 ===")
    
    # 创建解析器实例
    object_parser = SimpleObjectParser(
        chunk_size=args.chunk_size, 
        overlap_size=args.overlap_size,
    )
    
    # 显示已注册的解析器
    print(f"已注册的解析器: {object_parser.get_parser_info()}")
    
    results = []
    
    if args.test:
        # 运行默认测试案例
        print("\n运行默认测试案例...")
        test_files = [
            ("test", "pdf1.pdf"),
            ("test", "demo/不屈的精神.docx"),
            ("test", "demo/demo2/中华人民共和国刑法.txt")
        ]
        
        for bucket_name, object_path in test_files:
            print(f"\n=== 解析文档: {bucket_name}/{object_path} ===")
            result = object_parser.parse_object(bucket_name, object_path)
            results.append(result)
            
            # 显示解析结果
            print(f"文档名称: {result['doc_name']}")
            print(f"文档路径: {result['doc_path_name']}")
            print(f"文档类型: {result['doc_type']}")
            print(f"文档MD5: {result['doc_md5']}")
            print(f"文档大小: {result['doc_length']} 字节")
            print(f"文本片段数量: {len(result['contents'])}")
            
            if 'error' in result:
                print(f"错误信息: {result['error']}")
            elif result['contents']:
                # 显示所有片段的信息
                for content in result['contents']:
                    print(f"*************片段 (长度: {len(content)} 字符)")
                    print(f" {content}")
            
            print("-" * 100)
    
    elif args.file:
        # 解析指定的文件
        for file_path in args.file:
            print(f"\n=== 解析文档: {args.bucket}/{file_path} ===")
            result = object_parser.parse_object(args.bucket, file_path)
            results.append(result)
            
            # 显示解析结果
            print(f"文档名称: {result['doc_name']}")
            print(f"文档路径: {result['doc_path_name']}")
            print(f"文档类型: {result['doc_type']}")
            print(f"文档MD5: {result['doc_md5']}")
            print(f"文档大小: {result['doc_length']} 字节")
            print(f"文本片段数量: {len(result['contents'])}")
            
            if 'error' in result:
                print(f"错误信息: {result['error']}")
            elif result['contents']:
                # 显示第一个片段的信息
                for content in result['contents']:
                    print(f"*************片段 (长度: {len(content)} 字符)")
                    print(f" {content}")
            
            print("-" * 100)
    
    else:
        # 显示使用说明
        parser.print_help()
        print("\n使用示例:")
        print("  python simple_object_parser.py --test")
        print("  python simple_object_parser.py --bucket test --file test/demo/GoogleIO.pdf")
        print("  python simple_object_parser.py --bucket test --file test/demo/不屈的精神.docx")
    
    
    print(f"\n=== 处理完成 ===")
