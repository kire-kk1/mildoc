import argparse

from logger.logging import setup_logging

logger = setup_logging()
    


def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="Minio文档处理系统 - 将Minio中的文档解析并存储到Milvus向量数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 全量刷新模式
  python main.py --provider minio --mode full-refresh
  
  # 排查补漏模式
  python main.py --provider minio --mode backfill
  
  # 增量更新模式（实时监听）
  python main.py --provider minio --mode listen

  # 使用 OSS 作为对象存储提供商
  python main.py --provider oss --mode listen
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["full-refresh", "backfill", "listen"],
        required=True,
        help="运行模式选择: full-refresh=全量刷新, backfill=排查补漏, listen=增量更新(实时监听)"
    )
    
    parser.add_argument(
        "--provider",
        choices=["oss", "minio"],
        default="minio",
        type=str,
        help="对象存储提供商: oss=阿里云OSS, minio=Minio"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    logger.info("=== Minio文档处理系统 ===")
    logger.info(f"运行模式: {args.mode}")
    
    try:
        # 创建监听器实例
        if args.provider == "oss":
            from oss_event_handler import OSSEventHandler
            listener = OSSEventHandler()
            logger.info(f"使用 OSS 最为对象存储提供商")
        else:
            from minio_event_handler import MinioEventHandler
            listener = MinioEventHandler()
            logger.info(f"默认使用 Minio 作为对象存储提供商")
        
        logger.info("=== 系统初始化完成 ===")
        
        # 根据模式执行相应操作
        if args.mode == "full-refresh":
            logger.info("\n执行全量刷新模式...")
            listener.full_update()
            
        elif args.mode == "backfill":
            logger.info("\n执行排查补漏模式...")
            listener.backfill_update()
            
        elif args.mode == "listen":
            logger.info("\n执行增量更新模式（实时监听）...")
            logger.info("提示: 使用 Ctrl+C 停止监听，或使用 nohup 在后台运行")
            listener.start_listening()
        else:
            ## 使用方式说明
            logger.info("""
使用示例:
  # 全量刷新模式
  python main.py --provider minio --mode full-refresh
  
  # 排查补漏模式
  python main.py --provider minio --mode backfill
  
  # 增量更新模式（实时监听）
  python main.py --provider minio --mode listen

  # 使用 OSS 作为对象存储提供商
  python main.py --provider oss --mode listen            
            """)


        logger.info("\n程序执行完成")
        
    except KeyboardInterrupt:
        logger.info("\n用户中断程序")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        exit(1)


if __name__ == "__main__":
    main()
