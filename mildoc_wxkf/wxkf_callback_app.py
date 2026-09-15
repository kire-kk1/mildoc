"""
企业微信回调服务端
功能：接收企业微信的回调验证和消息推送

作者：开发工程师
日期：2025年01月
"""

import os
import logging
import json
from urllib.parse import unquote
from flask import Flask, request, abort
from WXBizMsgCrypt import WXBizMsgCrypt
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

def get_wecom_config():
    """获取企业微信配置，优先从环境变量获取"""
    corp_id = Config.CORP_ID
    token = Config.TOKEN
    encoding_aes_key = Config.ENCODING_AES_KEY
    
    if not token:
        logger.error("缺少TOKEN配置，请设置环境变量TOKEN或修改代码中的TOKEN变量")
        return None, None, None
    
    if not encoding_aes_key:
        logger.error("缺少ENCODING_AES_KEY配置，请设置环境变量WECOM_ENCODING_AES_KEY或修改代码中的ENCODING_AES_KEY变量")
        return None, None, None
    
    return corp_id, token, encoding_aes_key


@app.before_request
def log_request_info():
    """记录请求信息"""
    # 记录请求的基本信息
    logger.info(f"=== 收到HTTP请求 ===")
    logger.info(f"请求方法: {request.method}")
    logger.info(f"请求URL: {request.url}")
    logger.info(f"请求路径: {request.path}")
    logger.info(f"客户端IP: {request.remote_addr}")
    
    # 记录请求头
    headers = dict(request.headers)
    logger.info(f"请求头: {json.dumps(headers, indent=2, ensure_ascii=False)}")
    
    # 记录查询参数
    if request.args:
        args = dict(request.args)
        logger.info(f"查询参数: {json.dumps(args, indent=2, ensure_ascii=False)}")
    
    # 记录请求体（仅对POST/PUT等方法）
    if request.method in ['POST', 'PUT', 'PATCH']:
        try:
            if request.content_type and 'application/json' in request.content_type:
                # JSON数据
                json_data = request.get_json()
                if json_data:
                    logger.info(f"请求JSON: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
            else:
                # 原始数据
                raw_data = request.get_data(as_text=True)
                if raw_data:
                    logger.info(f"请求体: {raw_data[:500]}...")  # 限制长度避免日志过长
        except Exception as e:
            logger.warning(f"读取请求体时出错: {e}")

@app.after_request
def log_response_info(response):
    """记录响应信息"""
    logger.info(f"=== HTTP响应 ===")
    logger.info(f"响应状态码: {response.status_code}")
    logger.info(f"响应状态: {response.status}")
    
    # 记录响应头
    headers = dict(response.headers)
    logger.info(f"响应头: {json.dumps(headers, indent=2, ensure_ascii=False)}")
    
    # 记录响应内容
    try:
        if response.content_type and 'application/json' in response.content_type:
            # JSON响应
            logger.info(f"响应JSON: {response.get_data(as_text=True)}")
        else:
            # 其他类型响应
            response_data = response.get_data(as_text=True)
            if response_data:
                if len(response_data) > 500:
                    logger.info(f"响应内容: {response_data[:500]}...")
                else:
                    logger.info(f"响应内容: {response_data}")
            else:
                logger.info("响应内容: 无内容")
    except Exception as e:
        logger.warning(f"读取响应内容时出错: {e}")
    
    logger.info(f"=== 请求处理完成 ===\n")
    return response


def get_wxcrypt():
    """
    获取企业微信加解密工具对象
    """
    corp_id, token, encoding_aes_key = get_wecom_config()
    if not all([corp_id, token, encoding_aes_key]):
        logger.error("企业微信配置不完整")
        abort(500)
    return WXBizMsgCrypt(token, encoding_aes_key, corp_id)


@app.route('/callback/command', methods=['GET'])
def wecom_callback_get():
    """
    企业微信回调接口
    GET请求：用于验证回调URL的有效性
    """
    # 获取加解密工具对象
    wxcrypt = get_wxcrypt()
    
    # 获取URL参数
    msg_signature = request.args.get('msg_signature', '')
    timestamp = request.args.get('timestamp', '')
    nonce = request.args.get('nonce', '')
    
    logger.info(f"收到回调请求 - Method: {request.method}, msg_signature: {msg_signature}, timestamp: {timestamp}, nonce: {nonce}")
    
    # URL验证
    echostr = request.args.get('echostr', '')
    if not echostr:
        logger.error("GET请求缺少echostr参数")
        abort(400)
    
    if not all([msg_signature, timestamp, nonce]):
        logger.error("GET请求缺少必要参数")
        abort(400)
    
    try:
        
        # URL解码echostr参数
        echostr = unquote(echostr)
        logger.info(f"开始验证URL - echostr: {echostr[:50]}...")
        
        logger.info(f"echostr: {echostr}")

        # 验证URL并解密echostr
        result = wxcrypt.VerifyURL(msg_signature, timestamp, nonce, echostr)
        
        logger.info(f"验证URL结果: {result}")

        # 检查返回值类型
        if isinstance(result, tuple):
            ret, reply_echostr = result
        else:
            ret = result
            reply_echostr = None
        
        logger.info(f"验证结果 - 返回码: {ret}")
        if ret != 0:
            logger.error(f"URL验证失败，错误码: {ret}")
            if ret == -40001:
                logger.error("签名验证失败 - 请检查Token配置是否与企业微信后台一致")
            elif ret == -40002:
                logger.error("AES解密失败或CorpID不匹配 - 请检查EncodingAESKey和CorpID配置")
            else:
                logger.error(f"未知错误码: {ret}")
            abort(403)
        
        logger.info("URL验证成功")
        return reply_echostr
    except Exception as e:
        logger.error(f"URL验证过程发生错误: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        abort(500)
        
    



@app.route('/callback/command', methods=['POST'])
def wecom_callback_post():
    """
    企业微信回调接口
    POST请求：接收企业微信推送的消息和事件
    """
    # 获取加解密工具对象
    wxcrypt = get_wxcrypt()
    
    # 获取URL参数
    msg_signature = request.args.get('msg_signature', '')
    timestamp = request.args.get('timestamp', '')
    nonce = request.args.get('nonce', '')
    
    logger.info(f"收到回调请求 - Method: {request.method}, msg_signature: {msg_signature}, timestamp: {timestamp}, nonce: {nonce}")
    
    # 接收消息
    if not all([msg_signature, timestamp, nonce]):
        logger.error("POST请求缺少必要参数")
        abort(400)
        
    try:
        # 获取POST数据
        post_data = request.get_data(as_text=True)
        if not post_data:
            logger.error("POST请求体为空")
            abort(400)
        
        logger.info(f"收到POST消息: {post_data[:200]}...")
        
        # 解密消息
        ret, msg = wxcrypt.DecryptMsg(post_data, msg_signature, timestamp, nonce)
        
        if ret != 0:
            logger.error(f"消息解密失败，错误码: {ret}")
            abort(403)
        
        logger.info(f"消息解密成功: {msg}")
        
        # 处理消息（这里可以根据业务需求进行扩展）
        response_msg = handle_message(msg)
        
        if response_msg:
            # 加密响应消息
            ret, encrypted_msg = wxcrypt.EncryptMsg(response_msg, nonce, timestamp)
            if ret == 0:
                logger.info("响应消息加密成功")
                return encrypted_msg
            else:
                logger.error(f"响应消息加密失败，错误码: {ret}")
        
        # 返回空字符串表示成功接收但不回复
        return ''
        
    except Exception as e:
        logger.error(f"处理POST请求时发生错误: {str(e)}")
        abort(500)
    

def handle_message(msg):
    """
    处理解密后的消息
    
    Args:
        msg: 解密后的XML消息
        
    Returns:
        str: 要回复的消息（XML格式），如果不需要回复则返回None
    """
    try:
        import xml.etree.ElementTree as ET
        import time
        
        # 解析XML消息
        root = ET.fromstring(msg)
        msg_type = root.find('MsgType').text if root.find('MsgType') is not None else ''
        msg_id = root.find('MsgId').text if root.find('MsgId') is not None else ''
        
        logger.info(f"处理消息 - 类型: {msg_type}, 消息ID: {msg_id}")
        
        # 根据消息类型进行处理
        if msg_type == 'event':
            # 处理事件消息
            event = root.find('Event').text if root.find('Event') is not None else ''
            logger.info(f"收到事件: {event}")
            
            return process_event_message(event, root)
                
        # 对于其他类型的消息，记录日志但不回复
        logger.info(f"收到其他类型消息，暂不处理: {msg_type}")
        return None
        
    except Exception as e:
        logger.error(f"处理消息时发生错误: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return None



def process_event_message(event, root):
    """
    处理事件消息
    
    Args:
        event: 事件类型
        from_user: 发送者
        to_user: 接收者
        root: XML根节点
        
    Returns:
        str: 回复消息，如果不需要回复则返回None
    """
    if event == 'kf_msg_or_event':
        # 微信客服消息或事件
        logger.info("收到微信客服事件，开始处理客服消息")
        
        # 获取Token和OpenKfId
        token = root.find('Token').text if root.find('Token') is not None else ''
        open_kfid = root.find('OpenKfId').text if root.find('OpenKfId') is not None else ''
        
        if token and open_kfid:
            # 异步处理客服消息（避免阻塞回调响应）
            import threading
            
            def process_kf_messages():
                try:
                    # 延迟导入避免循环导入
                    from kf_message_handler import KfMessageHandler
                    kf_handler = KfMessageHandler()
                    kf_handler.process_kf_event(token, open_kfid)
                except Exception as e:
                    logger.error(f"处理客服消息异常: {e}")
            
            # 启动后台线程处理
            thread = threading.Thread(target=process_kf_messages)
            thread.daemon = True
            thread.start()
            
            logger.info(f"已启动客服消息处理线程 - OpenKfId: {open_kfid}")
        else:
            logger.error("客服事件缺少必要参数 - Token或OpenKfId为空")
        
        # 客服事件不需要回复
        return None
    
    # 其他事件类型
    logger.info(f"收到其他事件类型: {event}")
    return None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return {'status': 'ok', 'message': '企业微信回调服务运行正常'}

@app.route('/', methods=['GET'])
def index():
    return '''
    <h1>企业微信回调服务</h1>
    <p>服务正在运行...</p>
    '''



if __name__ == '__main__':
    # 检查配置
    corp_id, token, encoding_aes_key = get_wecom_config()
    
    if not all([corp_id, token, encoding_aes_key]):
        logger.error("配置不完整，请检查企业微信相关配置")
        logger.info("请设置以下环境变量或修改代码中的配置:")
        logger.info("- WECOM_CORP_ID: 企业ID")
        logger.info("- WECOM_TOKEN: 应用Token")
        logger.info("- WECOM_ENCODING_AES_KEY: 应用EncodingAESKey")
        exit(1)
    
    logger.info("企业微信回调服务启动中...")
    logger.info(f"企业ID: {corp_id}")
    logger.info(f"Token: {token[:10]}..." if token else "Token: 未配置")
    logger.info(f"EncodingAESKey: {encoding_aes_key[:10]}..." if encoding_aes_key else "EncodingAESKey: 未配置")

    # 启动服务
    port = Config.PORT
    logger.info(f"启动服务 - 端口: {port}")
    host = Config.HOST
    logger.info(f"启动服务 - 主机: {host}")
    debug = Config.DEBUG
    logger.info(f"启动服务 - 调试模式: {debug}")

    app.run(host=host, port=port, debug=debug, threaded=True, processes=1) 