#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
微信客服消息cursor持久化管理模块
用于管理每个客服账号的消息拉取cursor，避免重复处理消息

作者：开发工程师
日期：2025年01月
"""

import os
import json
import logging
import sqlite3
import threading
from typing import Dict, Optional
from config import Config

logger = logging.getLogger(__name__)

class CursorManager:
    """Cursor持久化管理器"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.DATABASE_PATH
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建cursor存储表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS kf_cursors (
                        open_kfid TEXT PRIMARY KEY,
                        cursor TEXT NOT NULL,
                        last_updated INTEGER NOT NULL,
                        message_count INTEGER DEFAULT 0,
                        created_time INTEGER DEFAULT (strftime('%s', 'now'))
                    )
                ''')
                
                # 创建消息去重表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processed_messages (
                        msgid TEXT PRIMARY KEY,
                        open_kfid TEXT NOT NULL,
                        external_userid TEXT,
                        msgtype TEXT,
                        origin INTEGER,
                        processed_time INTEGER DEFAULT (strftime('%s', 'now')),
                        reply_sent INTEGER DEFAULT 0
                    )
                ''')
                
                # 创建索引
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_processed_messages_time 
                    ON processed_messages(processed_time)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_processed_messages_kfid 
                    ON processed_messages(open_kfid)
                ''')
                
                conn.commit()
                logger.info("Cursor管理数据库初始化完成")
                
        except Exception as e:
            logger.error(f"初始化cursor数据库失败: {e}")
    
    def get_cursor(self, open_kfid: str) -> str:
        """
        获取指定客服账号的cursor
        
        Args:
            open_kfid: 客服账号ID
            
        Returns:
            cursor字符串，如果不存在则返回空字符串
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT cursor FROM kf_cursors WHERE open_kfid = ?',
                        (open_kfid,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        logger.debug(f"获取cursor成功 - {open_kfid}: {result[0][:20]}...")
                        return result[0]
                    else:
                        logger.info(f"客服账号 {open_kfid} 无历史cursor，将进行首次拉取")
                        return ""
                        
        except Exception as e:
            logger.error(f"获取cursor失败: {e}")
            return ""
    
    def save_cursor(self, open_kfid: str, cursor: str, message_count: int = 0) -> bool:
        """
        保存指定客服账号的cursor
        
        Args:
            open_kfid: 客服账号ID
            cursor: 新的cursor值
            message_count: 本次处理的消息数量
            
        Returns:
            保存是否成功
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    db_cursor = conn.cursor()
                    
                    # 使用UPSERT语法
                    db_cursor.execute('''
                        INSERT OR REPLACE INTO kf_cursors 
                        (open_kfid, cursor, last_updated, message_count)
                        VALUES (?, ?, strftime('%s', 'now'), 
                               COALESCE((SELECT message_count FROM kf_cursors WHERE open_kfid = ?), 0) + ?)
                    ''', (open_kfid, cursor, open_kfid, message_count))
                    
                    conn.commit()
                    logger.info(f"保存cursor成功 - {open_kfid}: {cursor[:20]}..., 消息数: {message_count}")
                    return True
                    
        except Exception as e:
            logger.error(f"保存cursor失败: {e}")
            return False
    
    def is_message_processed(self, msgid: str) -> bool:
        """
        检查消息是否已经处理过
        
        Args:
            msgid: 消息ID
            
        Returns:
            是否已处理
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT 1 FROM processed_messages WHERE msgid = ?',
                        (msgid,)
                    )
                    result = cursor.fetchone()
                    return result is not None
                    
        except Exception as e:
            logger.error(f"检查消息处理状态失败: {e}")
            return False
    
    def mark_message_processed(self, msgid: str, open_kfid: str, external_userid: str = "", 
                             msgtype: str = "", origin: int = 0, reply_sent: bool = False) -> bool:
        """
        标记消息为已处理
        
        Args:
            msgid: 消息ID
            open_kfid: 客服账号ID
            external_userid: 外部用户ID
            msgtype: 消息类型
            origin: 消息来源
            reply_sent: 是否已发送回复
            
        Returns:
            标记是否成功
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO processed_messages 
                        (msgid, open_kfid, external_userid, msgtype, origin, reply_sent)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (msgid, open_kfid, external_userid, msgtype, origin, int(reply_sent)))
                    
                    conn.commit()
                    logger.debug(f"标记消息已处理: {msgid}")
                    return True
                    
        except Exception as e:
            logger.error(f"标记消息处理状态失败: {e}")
            return False
    
    def cleanup_old_records(self, days: int = 30) -> bool:
        """
        清理旧的处理记录
        
        Args:
            days: 保留天数
            
        Returns:
            清理是否成功
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 清理旧的消息处理记录
                    cursor.execute('''
                        DELETE FROM processed_messages 
                        WHERE processed_time < strftime('%s', 'now', '-{} days')
                    '''.format(days))
                    
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    if deleted_count > 0:
                        logger.info(f"清理了 {deleted_count} 条旧的消息处理记录")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"清理旧记录失败: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        获取cursor管理统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 获取客服账号数量
                    cursor.execute('SELECT COUNT(*) FROM kf_cursors')
                    kf_count = cursor.fetchone()[0]
                    
                    # 获取总消息处理数量
                    cursor.execute('SELECT SUM(message_count) FROM kf_cursors')
                    total_messages = cursor.fetchone()[0] or 0
                    
                    # 获取今日处理消息数量
                    cursor.execute('''
                        SELECT COUNT(*) FROM processed_messages 
                        WHERE processed_time >= strftime('%s', 'now', 'start of day')
                    ''')
                    today_messages = cursor.fetchone()[0]
                    
                    # 获取已回复消息数量
                    cursor.execute('SELECT COUNT(*) FROM processed_messages WHERE reply_sent = 1')
                    replied_messages = cursor.fetchone()[0]
                    
                    return {
                        'kf_accounts': kf_count,
                        'total_messages': total_messages,
                        'today_messages': today_messages,
                        'replied_messages': replied_messages,
                        'reply_rate': round(replied_messages / max(total_messages, 1) * 100, 2)
                    }
                    
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def get_kf_account_info(self, open_kfid: str) -> Optional[Dict]:
        """
        获取指定客服账号的详细信息
        
        Args:
            open_kfid: 客服账号ID
            
        Returns:
            账号信息字典
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 获取cursor信息
                    cursor.execute('''
                        SELECT cursor, last_updated, message_count, created_time 
                        FROM kf_cursors WHERE open_kfid = ?
                    ''', (open_kfid,))
                    cursor_info = cursor.fetchone()
                    
                    if not cursor_info:
                        return None
                    
                    # 获取今日消息数量
                    cursor.execute('''
                        SELECT COUNT(*) FROM processed_messages 
                        WHERE open_kfid = ? AND processed_time >= strftime('%s', 'now', 'start of day')
                    ''', (open_kfid,))
                    today_count = cursor.fetchone()[0]
                    
                    return {
                        'open_kfid': open_kfid,
                        'cursor': cursor_info[0][:20] + '...' if cursor_info[0] else '',
                        'last_updated': cursor_info[1],
                        'total_messages': cursor_info[2],
                        'today_messages': today_count,
                        'created_time': cursor_info[3]
                    }
                    
        except Exception as e:
            logger.error(f"获取客服账号信息失败: {e}")
            return None

# 全局cursor管理器实例
cursor_manager = CursorManager() 