#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mildoc 企业知识库 —— 性能与质量评测脚本
================================================
测量 5 项指标：
  ① 端到端延迟（P50 / P95 / P99）   ② LLM 首 Token 延迟
  ③ Recall@Top-3 命中率            ④ 幻觉率
  ⑤ 稳定运行天数
运行方式：cd ~/mildoc_202601/mildoc_wxkf && uv run python benchmark.py
"""

import time          # 计时用
import os            # 文件路径用
import statistics    # 算中位数/分位数用
from datetime import datetime  # 算运行天数用

# 导入项目自带的两个入口函数（本文件与 rag_service.py 同目录，可直接导入）
from rag_service import query_question, get_rag_service

# 评测集文件路径（和本脚本同目录）
QA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_pairs.txt")
# wxkf 日志路径（算稳定运行天数用）
LOG_FILE = os.path.expanduser("~/mildoc_202601/mildoc_wxkf/mildoc_wxkf.log")

# 幻觉测试专用问题（这些都不在知识库里，用于看模型会不会"瞎编"）
HALLUCINATION_QUESTIONS = [
    "推荐一部好看的科幻电影",
    "今天烟台的天气怎么样",
    "帮我写一首赞美大海的诗",
    "1+1等于几？",
    "讲一个笑话",
    "世界上最高的山峰有多高",
    "推荐一家好吃的火锅店",
    "2026年世界杯冠军是谁",
]

# 拒答关键词：回答里出现这些词 = 模型正确承认"不知道"（不算幻觉）
REJECT_KEYWORDS = ["抱歉", "无法", "没有找到", "未找到", "不知道", "不在",
                   "无法回答", "没有相关信息", "暂不支持", "知识库", "不属于"]


def load_qa_pairs():
    """读取评测集文件（qa_pairs.txt）
    每行格式：问题|期望命中的文档关键词|期望回答中的关键词（第3列可省略）
    # 开头的是注释，自动跳过
    """
    pairs = []
    with open(QA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                pairs.append({"question": parts[0], "doc_kw": parts[1],
                              "ans_kw": parts[2] if len(parts) > 2 else ""})
    return pairs


def warmup():
    """预热：第一次调用 query_question 会初始化 RAG 服务
    （加载模型配置、连接 Milvus、加载向量集合），耗时几十秒很正常。
    预热结果不计入延迟，避免拉高 P95。
    """
    t0 = time.time()
    query_question("你好")
    print(f"[预热] RAG 服务初始化完成，耗时 {time.time()-t0:.1f}s\n")


def percentile(data, p):
    """计算百分位数：把数据排序后取第 p% 位置的值
    例如 p=95 → P95 延迟（95% 的请求快于这个值）
    """
    s = sorted(data)
    idx = min(len(s) - 1, int(len(s) * p / 100))
    return s[idx]


def test_latency(pairs):
    """① 端到端延迟：逐条问题完整跑一遍 RAG 链路（检索+重排+生成），记录耗时"""
    print("=" * 60)
    print("[1/4] 端到端延迟测试中（每条问题完整跑检索+重排+生成）...")
    latencies = []
    for i, p in enumerate(pairs, 1):
        q = p["question"]
        t0 = time.time()
        try:
            resp = query_question(q)          # 调用完整 RAG 问答
            dt = time.time() - t0
            latencies.append(dt)
            print(f"  [{i}/{len(pairs)}] {dt:.2f}s  {q[:30]}")
        except Exception as e:
            print(f"  [{i}/{len(pairs)}] 失败: {e}")
    if latencies:
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        p99 = percentile(latencies, 99)
        print(f"\n  ➜ 端到端延迟：P50={p50:.2f}s  P95={p95:.2f}s  P99={p99:.2f}s")
        print(f"  ➜ 测试 {len(latencies)} 条，最快 {min(latencies):.2f}s，最慢 {max(latencies):.2f}s")
    return latencies


def test_ttft():
    """② LLM 首 Token 延迟：用流式接口直连千问，测"发出请求→收到第一个字"的耗时"""
    print("=" * 60)
    print("[2/4] LLM 首 Token 延迟测试（Qwen-Plus 流式直连）...")
    try:
        from langchain_openai import ChatOpenAI
        from rag_service import get_rag_service
        svc = get_rag_service()          # 拿已初始化的 RAG 服务实例
        llm = svc.llm if svc else None   # 取出里面的 LLM 对象
        if llm is None:
            print("  ➜ 拿不到 LLM 实例，跳过")
            return
        # 从现有 LLM 对象反向读取配置（不猜字段名）
        api_key = getattr(llm, "openai_api_key", None) or getattr(llm, "api_key", None)
        api_base = getattr(llm, "openai_api_base", None) or getattr(llm, "openai_api_base", None)
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "qwen-plus")
        # 用读到的配置重建一个流式 LLM（RAG 内部那个是非流式的）
        stream_llm = ChatOpenAI(
            model=model_name, api_key=api_key, base_url=api_base,
            streaming=True, temperature=0.1,
        )
        ttf = []
        for _ in range(3):               # 测 3 次取平均
            t0 = time.time()
            first = None
            for chunk in stream_llm.stream("请用一句话介绍你自己"):
                if first is None and chunk.content:
                    first = time.time() - t0
            if first:
                ttf.append(first)
        if ttf:
            avg = statistics.mean(ttf)
            print(f"  ➜ 首 Token 延迟（LLM 层）：平均 {avg:.2f}s，最小 {min(ttf):.2f}s")
    except Exception as e:
        print(f"  ➜ 首 Token 测试失败（不影响其他指标）: {e}")


def test_recall(pairs):
    """③ Recall@Top-3：对每个问题做向量检索，看期望文档是否出现在 Top-3 里
    注意：get_similar_documents 是 RAGService 类的方法，需要先拿实例再调用
    """
    print("=" * 60)
    print("[3/4] Recall@Top-3 检索命中测试...")
    service = get_rag_service()          # 获取 RAG 服务实例（单例）
    if service is None:
        print("  ➜ RAG 服务初始化失败，跳过 Recall 测试")
        return 0
    hit = 0
    total = 0
    for i, p in enumerate(pairs, 1):
        q, kw = p["question"], p["doc_kw"]
        total += 1
        try:
            docs = service.get_similar_documents(q, top_k=3)   # 调实例方法
            names = []
            for d in docs:
                # 兼容 dict 或 Milvus Hit 对象两种返回结构
                if isinstance(d, dict):
                    name = d.get("doc_name") or d.get("doc_path_name") or str(d)
                else:
                    name = getattr(d, "entity", d)
                names.append(str(name))
            ok = any(kw in n for n in names)   # 期望关键词是否出现在任一文档名
            hit += 1 if ok else 0
            print(f"  [{i}/{total}] {'✓' if ok else '✗'} {q[:25]}  → Top3: {[n[:20] for n in names]}")
        except Exception as e:
            print(f"  [{i}/{total}] 检索失败: {e}")
    recall = hit / total if total else 0
    print(f"\n  ➜ Recall@Top-3 = {recall*100:.1f}% （{hit}/{total}）")
    return recall


def test_hallucination():
    """④ 幻觉率：问 8 个知识库外的问题
    正确行为 = 回答里出现拒答词（承认不知道）；一本正经给答案 = 疑似幻觉
    输出后请人工复核一遍，看标注是否合理
    """
    print("=" * 60)
    print("[4/4] 幻觉率测试（8 个知识库外问题）...")
    hallucinated = 0
    for i, q in enumerate(HALLUCINATION_QUESTIONS, 1):
        try:
            resp = query_question(q)
            answer = (resp.content or "").strip()[:60]
            rejected = any(k in answer for k in REJECT_KEYWORDS)
            if not rejected:
                hallucinated += 1
            print(f"  [{i}/8] {'疑似幻觉' if not rejected else '正常拒答'} 问:{q} 答:{answer}")
        except Exception as e:
            print(f"  [{i}/8] 失败: {e}")
    print(f"\n  ➜ 疑似幻觉率 = {hallucinated/len(HALLUCINATION_QUESTIONS)*100:.0f}% "
          f"（{hallucinated}/{len(HALLUCINATION_QUESTIONS)}，请人工复核上面标注）")


def calc_uptime():
    """⑤ 稳定运行天数：扫描日志前50行找最早时间戳，算到现在的天数"""
    print("=" * 60)
    print("[5] 稳定运行天数统计...")
    import re                                   # 正则：匹配 2026-09-12 20:47:11 这种格式
    pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    log_file = LOG_FILE
    try:
        if not os.path.exists(log_file):
            alt = os.path.expanduser("~/mildoc_202601/mildoc_index/mildoc_index.log")
            if os.path.exists(alt):
                log_file = alt
        first_time = None
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(50):                 # 只看前 50 行
                line = f.readline()
                if not line:
                    break
                m = pattern.search(line)
                if m:                           # 找到第一个时间戳
                    first_time = datetime.strptime(m.group(), "%Y-%m-%d %H:%M:%S")
                    ts_str = m.group()
                    break
        if first_time is None:
            print("  ➜ 日志前50行没找到时间戳，手动看日志即可")
            return 0
        days = (datetime.now() - first_time).total_seconds() / 86400
        print(f"  ➜ 日志最早记录: {ts_str}，稳定运行约 {days:.1f} 天")
        return days
    except Exception as e:
        print(f"  ➜ 计算失败（手动看日志即可）: {e}")
        return 0

def main():
    """主流程：按顺序跑全部测试"""
    print("Mildoc 企业知识库评测脚本启动")
    print(f"评测集文件: {QA_FILE}")
    if not os.path.exists(QA_FILE):
        print("错误：找不到 qa_pairs.txt！请先创建评测集文件（见模板）。")
        sys.exit(1)

    pairs = load_qa_pairs()
    print(f"读取到 {len(pairs)} 条评测问题\n")

    warmup()              # 先初始化（不计时）
    latencies = test_latency(pairs)   # ① 延迟
    test_ttft()           # ② 首 Token
    test_recall(pairs)    # ③ Recall
    test_hallucination()  # ④ 幻觉
    calc_uptime()         # ⑤ 运行天数

    print("\n" + "=" * 60)
    print("评测完成！把上面 ➜ 开头的结果抄进简历即可。")
    print("=" * 60)


if __name__ == "__main__":
    main()