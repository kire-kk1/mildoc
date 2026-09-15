import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
from logger.logging import setup_logging

load_dotenv()

logger = setup_logging()

class EmbeddingTool:
    def __init__(self):
        """
        初始化embedding工具
        """
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.model = os.getenv("ENBEDDING_MODEL")
        self.dimensions = int(os.getenv("MILVUS_VECTOR_DIM"))
        self.encoding_format = "float"

        if not self.client or not self.model or not self.dimensions or not self.encoding_format:
            logger.error("Embedding工具初始化失败")
            raise ValueError("Embedding工具初始化失败")
        
    def get_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的embedding向量
        
        Args:
            text (str): 输入文本
            
        Returns:
            List[float]: 向量列表
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.dimensions,
                encoding_format=self.encoding_format
            )
            
            if not response.data:
                logger.error("获取embedding失败")
                return []
            
            embedding = response.data[0].embedding
            if not embedding:
                logger.error("获取embedding失败")
                return []
            
            if len(embedding) != self.dimensions:
                logger.error("获取embedding失败")
                return []
            
            return embedding
            
        except Exception as e:
            print(f"获取embedding失败: {e}")
            return []
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取多个文本的embedding向量
        
        Args:
            texts (List[str]): 文本列表
            
        Returns:
            List[List[float]]: 向量列表的列表
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format=self.encoding_format
            )
            
            if not response.data:
                logger.error("批量获取embedding失败")
                return []
            
            embeddings = [data.embedding for data in response.data] 
            
            if len(embeddings) != len(texts):
                logger.error("批量获取embedding失败")
                return []

            for embedding in embeddings:
                if len(embedding) != self.dimensions:
                    logger.error("批量获取embedding失败")
                    return []
            
            return embeddings
            
        except Exception as e:
            print(f"批量获取embedding失败: {e}")
            return []
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            dict: 模型信息
        """
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "encoding_format": self.encoding_format,
            "base_url": self.client.base_url
        }


# 使用示例和测试
if __name__ == "__main__":
    print("=== Embedding工具测试 ===")
    
    # 创建embedding工具实例
    embedding_tool = EmbeddingTool()
    
    # 显示模型信息
    print("模型信息:", embedding_tool.get_model_info())
    
    # 单个文本embedding测试
    print("\n=== 单个文本embedding测试 ===")
    test_text = "这是一个测试文档的内容，用于生成向量表示。"
    embedding = embedding_tool.get_embedding(test_text)
    if embedding:
        print(f"文本: {test_text}")
        print(f"向量维度: {len(embedding)}")
        print(f"向量前5个值: {embedding[:5]}")
    
    # 批量文本embedding测试
    print("\n=== 批量文本embedding测试 ===")
    test_texts = [
        "风急天高猿啸哀",
        "渚清沙白鸟飞回", 
        "无边落木萧萧下",
        "不尽长江滚滚来"
    ]
    
    embeddings = embedding_tool.get_embeddings_batch(test_texts)
    if embeddings:
        print(f"处理了 {len(embeddings)} 个文本")
        for i, (text, emb) in enumerate(zip(test_texts, embeddings)):
            print(f"文本{i+1}: {text}")
            print(f"向量维度: {len(emb)}, 前3个值: {emb[:5]}")
    
    print("\n=== 测试完成 ===")
