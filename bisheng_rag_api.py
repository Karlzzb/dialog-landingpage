import requests
import json
import os
import time
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 加载 .env（需要 python-dotenv；未安装时回退到进程环境变量）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _require_env(name):
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"缺少环境变量 {name}，请在 .env 中配置（参考 .env.example）")
    return val


# ============ 配置（密钥从环境变量读取，不入库）============
BASE_URL = os.getenv("BISHENG_BASE_URL", "http://10.30.186.171:3001")
# bisheng 用短期 JWT，过期需更新 BISHENG_ACCESS_TOKEN。
AUTH_COOKIE = f"lang=zh-Hans; access_token_cookie={_require_env('BISHENG_ACCESS_TOKEN')}"

LLM_CONFIG = {
    "api_key": _require_env("INGEST_LLM_API_KEY"),
    "base_url": os.getenv("INGEST_LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
    "model": os.getenv("INGEST_LLM_MODEL", "deepseek-v3-2-251201"),
    "temperature": 0.3,
    "timeout": 180,
    "max_retries": 2,
}

# ===== 固定业务配置（仅文件路径、知识库名称等不变）=====
KB_NAME = "我的文档库1"
KB_DESCRIPTION = "存储项目文档"
FILE_PATH = "/Users/xiexinyu/Desktop/AI/AI知识汇总.docx"
MODEL_ID = "12"

MAX_ANALYZE_CHUNKS = 5
MAX_ANALYZE_PAIRS = 4
# =====================================================


def create_llm():
    return ChatOpenAI(**LLM_CONFIG)


def read_file_text(file_path):
    """尝试读取文件内容为纯文本（用于正则提取分隔符）"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.md':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.docx':
        try:
            import docx
            doc = docx.Document(file_path)
            return '\n'.join([p.text for p in doc.paragraphs])
        except ImportError:
            print("⚠️ 请先安装 python-docx: pip install python-docx")
            return None
        except Exception as e:
            print(f"⚠️ 读取 docx 失败: {e}")
            return None
    else:
        print(f"⚠️ 不支持的文件类型 {ext}，无法自动提取文本用于正则匹配")
        return None


def interactive_config(file_path):
    """交互式获取切分参数"""
    print("\n" + "=" * 50)
    print("🔧 请配置文档切分参数")
    print("=" * 50)

    # 1. chunk_size
    while True:
        size_input = input("请输入 chunk_size（每个块的最大字符数，默认 1000）: ").strip()
        if not size_input:
            chunk_size = 1000
            break
        try:
            chunk_size = int(size_input)
            if chunk_size > 0:
                break
            print("❌ 必须为正整数，请重新输入")
        except ValueError:
            print("❌ 请输入有效数字")

    # 2. chunk_overlap
    while True:
        overlap_input = input("请输入 chunk_overlap（块间重叠字符数，默认 0）: ").strip()
        if not overlap_input:
            chunk_overlap = 0
            break
        try:
            chunk_overlap = int(overlap_input)
            if chunk_overlap >= 0 and chunk_overlap < chunk_size:
                break
            print(f"❌ 重叠必须 >=0 且小于 chunk_size ({chunk_size})")
        except ValueError:
            print("❌ 请输入有效数字")

    # 3. 分隔策略选择
    print("\n请选择分隔策略：")
    print("  1. 段落 + 换行     (\\n\\n, \\n)  【推荐】")
    print("  2. 仅段落          (\\n\\n)")
    print("  3. 仅换行          (\\n)")
    print("  4. 句号 + 换行     (。, \\n)")
    print("  5. 自定义分隔符    (手动输入，逗号分隔)")
    print("  6. 正则表达式提取  (读取文件，用正则匹配内容作为分隔符)")

    choice = input("请输入数字 (1-6): ").strip()
    while choice not in ['1', '2', '3', '4', '5', '6']:
        print("❌ 请输入 1~6 之间的数字")
        choice = input("请重新输入: ").strip()

    separator = []
    separator_rule = []

    if choice == '1':
        separator = ["\n\n", "\n"]
        separator_rule = ["after", "after"]
    elif choice == '2':
        separator = ["\n\n"]
        separator_rule = ["after"]
    elif choice == '3':
        separator = ["\n"]
        separator_rule = ["after"]
    elif choice == '4':
        separator = ["。", "\n"]
        separator_rule = ["after", "after"]
    elif choice == '5':
        custom_input = input("请输入自定义分隔符，多个用英文逗号分隔（例如: \\n\\n,\\n,。 ）: ").strip()
        if not custom_input:
            print("⚠️ 未输入，使用默认段落分隔")
            separator = ["\n\n"]
            separator_rule = ["after"]
        else:
            # 处理转义字符（如 \n）
            raw_parts = custom_input.split(',')
            separator = []
            for p in raw_parts:
                p = p.strip()
                if p:
                    # 将字符串字面量中的 \n 转为真正的换行符
                    p = p.encode('utf-8').decode('unicode_escape')
                    separator.append(p)
            separator_rule = ["after"] * len(separator)
    elif choice == '6':
        regex_pattern = input("请输入正则表达式（例如: (?:^|\\n)(一、|二、|三、) ）: ").strip()
        if not regex_pattern:
            print("⚠️ 未输入正则，使用默认段落分隔")
            separator = ["\n\n"]
            separator_rule = ["after"]
        else:
            print(f"⏳ 正在读取文件并提取匹配项...")
            content = read_file_text(file_path)
            if content is None:
                print("⚠️ 读取文件失败，回退到默认分隔符")
                separator = ["\n\n"]
                separator_rule = ["after"]
            else:
                matches = re.findall(regex_pattern, content, re.MULTILINE | re.DOTALL)
                # 去重，并过滤空字符串
                if isinstance(matches, list):
                    # 如果正则中有多个捕获组，matches 会是元组列表，需要展平
                    flat_matches = []
                    for m in matches:
                        if isinstance(m, tuple):
                            flat_matches.extend([x for x in m if x])
                        elif isinstance(m, str) and m:
                            flat_matches.append(m)
                    matches = list(set(flat_matches))
                else:
                    matches = list(set([m for m in matches if m]))

                if not matches:
                    print("⚠️ 正则未匹配到任何内容，回退到默认分隔符")
                    separator = ["\n\n"]
                    separator_rule = ["after"]
                else:
                    # 按长度降序排序，避免短分隔符被长分隔符包含（如 "一、" 和 "一、二、"）
                    matches.sort(key=len, reverse=True)
                    separator = matches
                    separator_rule = ["after"] * len(separator)
                    print(f"✅ 正则提取到 {len(separator)} 个分隔符: {separator[:10]}{'...' if len(separator)>10 else ''}")

    print("\n" + "-" * 30)
    print(f"✅ 最终参数：")
    print(f"   chunk_size = {chunk_size}")
    print(f"   chunk_overlap = {chunk_overlap}")
    print(f"   separator = {separator}")
    print(f"   separator_rule = {separator_rule}")
    print("-" * 30 + "\n")

    return chunk_size, chunk_overlap, separator, separator_rule


# ---------- 以下为 API 调用函数（与原脚本相同，略作精简） ----------

def create_knowledge_base(name, description="", model_id="12"):
    url = f"{BASE_URL}/api/v1/knowledge/create"
    payload = {"name": name, "description": description, "type": 0, "model": model_id}
    headers = {'Content-Type': 'application/json', 'Cookie': AUTH_COOKIE}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        if result.get('status_code') == 200:
            knowledge_id = result.get('data', {}).get('id')
            print(f"✅ 知识库创建成功，ID: {knowledge_id}")
            return knowledge_id
        else:
            print(f"❌ 业务错误: {result.get('status_message')}")
            return None
    else:
        print(f"❌ HTTP错误: {response.text}")
        return None


def upload_file(knowledge_id, file_path):
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    file_name = os.path.basename(file_path)
    print(f"📤 上传文件: {file_name} 到知识库 {knowledge_id}")
    url = f"{BASE_URL}/api/v1/knowledge/upload/{knowledge_id}"
    files = {'file': (file_name, open(file_path, 'rb'), 'application/octet-stream')}
    headers = {'Cookie': AUTH_COOKIE}
    response = requests.post(url, files=files, headers=headers)
    if response.status_code == 200:
        result = response.json()
        if result.get('status_code') == 200:
            presigned_url = result.get('data', {}).get('file_path')
            if presigned_url:
                print("✅ 文件上传成功")
                return presigned_url
            else:
                print("❌ 响应中无 file_path")
                return None
        else:
            print(f"❌ 业务错误: {result.get('status_message')}")
            return None
    else:
        print(f"❌ HTTP错误: {response.text}")
        return None


def preview_file(knowledge_id, file_path, separator, separator_rule, chunk_size, chunk_overlap,
                 retain_images=True, enable_formula=True, force_ocr=True,
                 header_start_row=1, header_end_row=1, slice_length=10, append_header=True):
    url = f"{BASE_URL}/api/v1/knowledge/preview"
    payload = {
        "cache": False,
        "knowledge_id": str(knowledge_id),
        "file_list": [{
            "file_path": file_path,
            "excel_rule": {
                "slice_length": slice_length,
                "append_header": append_header,
                "header_start_row": header_start_row,
                "header_end_row": header_end_row
            }
        }],
        "separator": separator,
        "separator_rule": separator_rule,
        "chunk_size": str(chunk_size),
        "chunk_overlap": str(chunk_overlap),
        "retain_images": retain_images,
        "enable_formula": enable_formula,
        "force_ocr": force_ocr,
        "fileter_page_header_footer": True
    }
    headers = {'Content-Type': 'application/json', 'Cookie': AUTH_COOKIE}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        if result.get('status_code') == 200:
            preview_file_id = result.get('data', {}).get('preview_file_id')
            print(f"✅ 预览任务提交成功，preview_file_id: {preview_file_id}")
            return preview_file_id
        else:
            print(f"❌ 业务错误: {result.get('status_message')}")
            return None
    else:
        print(f"❌ HTTP错误: {response.text}")
        return None


def get_preview_result(preview_file_id, max_attempts=10, interval=2):
    url = f"{BASE_URL}/api/v1/knowledge/preview/status"
    params = {"preview_file_id": preview_file_id}
    headers = {'Cookie': AUTH_COOKIE}
    print(f"\n⏳ 等待预览结果生成（最多 {max_attempts * interval} 秒）...")
    for i in range(max_attempts):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            result = response.json()
            if result.get('status_code') == 200:
                data = result.get('data', {})
                status = data.get('status')
                print(f"  [{i + 1}/{max_attempts}] 预览状态: {status}")
                if status == "completed":
                    print("✅ 预览完成！")
                    chunks_data = data.get('data', {})
                    chunks = chunks_data.get('chunks', [])
                    abstract = chunks_data.get('abstract', '')
                    return {
                        "chunks": chunks,
                        "abstract": abstract,
                        "parse_type": chunks_data.get('parse_type'),
                        "file_url": chunks_data.get('file_url')
                    }
                elif status == "failed":
                    print("❌ 预览失败")
                    return None
            else:
                print(f"  [{i + 1}/{max_attempts}] 业务错误: {result.get('status_message')}")
        else:
            print(f"  [{i + 1}/{max_attempts}] HTTP错误: {response.status_code}")
        time.sleep(interval)
    print("⏰ 预览等待超时")
    return None


def print_preview_result(preview_result):
    if not preview_result:
        print("❌ 无预览结果")
        return
    chunks = preview_result.get('chunks', [])
    abstract = preview_result.get('abstract', '')
    parse_type = preview_result.get('parse_type', '')
    print("\n" + "=" * 60)
    print("📄 文档预览结果")
    print("=" * 60)
    print(f"解析类型: {parse_type}")
    print(f"文档摘要: {abstract}")
    print(f"切分块数: {len(chunks)}")
    print("-" * 60)
    for idx, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        chunk_index = metadata.get('chunk_index', idx)
        print(f"\n【Chunk {chunk_index}】")
        print(text)
        print("-" * 40)


def process_file(knowledge_id, file_name, file_path, separator, separator_rule, chunk_size, chunk_overlap):
    url = f"{BASE_URL}/api/v1/knowledge/process"
    payload = {
        "knowledge_id": knowledge_id,
        "file_list": [{"file_id": 1, "file_name": file_name, "file_path": file_path}],
        "separator": separator,
        "separator_rule": separator_rule,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }
    headers = {'Content-Type': 'application/json', 'Cookie': AUTH_COOKIE}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        if result.get('status_code') == 200:
            data = result.get('data', [])
            if data:
                file_info = data[0]
                file_id = file_info.get('id')
                status = file_info.get('status')
                status_map = {5: "⏳ 处理中", 6: "✅ 处理成功", 7: "❌ 处理失败"}
                print(f"✅ 正式处理任务提交成功！文件ID: {file_id}, 状态: {status} ({status_map.get(status, '未知')})")
                return file_id
            else:
                print("⚠️ 响应中无文件数据")
                return None
        else:
            print(f"❌ 业务错误: {result.get('status_message')}")
            return None
    else:
        print(f"❌ HTTP错误: {response.text}")
        return None


# ---------- 整合自 123.py 的知识库列表查询函数 ----------
def list_knowledge_bases():
    """分页获取所有知识库列表，支持多种分页参数"""
    url = f"{BASE_URL}/api/v1/knowledge"
    headers = {'Cookie': AUTH_COOKIE}
    all_items = []
    total = 0

    # 尝试多种分页参数组合（按成功率高的优先）
    schemes = [
        ('page', 'page_size'),   # 常见
        ('page', 'size'),
        ('page', 'per_page'),
        ('offset', 'limit'),
    ]

    for offset_name, limit_name in schemes:
        print(f"⏳ 尝试分页方案: {offset_name}/{limit_name}")
        page = 1
        per_page = 100  # 请求大数量，但API可能限制最大10，没关系，只要循环能继续
        accumulated = []
        total = 0
        try:
            while True:
                params = {}
                if offset_name == 'page':
                    params[offset_name] = page
                else:
                    params[offset_name] = (page - 1) * per_page
                params[limit_name] = per_page

                resp = requests.get(url, headers=headers, params=params, timeout=30)
                if resp.status_code != 200:
                    break
                result = resp.json()
                if result.get('status_code') != 200:
                    break
                data = result.get('data', {})
                items = data.get('data', [])
                total = data.get('total', 0)

                if not items:
                    break  # 没有更多数据，结束循环

                accumulated.extend(items)

                # 如果已经获取了所有数据，退出循环
                if total > 0 and len(accumulated) >= total:
                    break

                # 继续下一页
                page += 1

            # 如果此方案获取的数据多于之前，则采用此结果
            if len(accumulated) > len(all_items):
                all_items = accumulated
                # 如果已经获取全部，终止尝试其他方案
                if total > 0 and len(all_items) >= total:
                    break
        except Exception as e:
            print(f"⚠️ 方案异常: {e}")
            continue

    # 如果所有分页方案都失败，尝试直接请求（不带参数）
    if not all_items:
        print("⏳ 所有分页方案均失败，尝试直接请求...")
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                if result.get('status_code') == 200:
                    all_items = result.get('data', {}).get('data', [])
                    total = result.get('data', {}).get('total', len(all_items))
        except Exception as e:
            print(f"❌ 直接请求失败: {e}")
            return []

    if not all_items:
        print("📭 没有找到任何知识库")
        return []

    print(f"\n📚 共 {total} 个知识库（成功获取 {len(all_items)} 个）：")
    print("-" * 80)
    for idx, item in enumerate(all_items, start=1):
        kb_id = item.get('id')
        name = item.get('name', '未命名')
        user_name = item.get('user_name', '未知用户')
        create_time = item.get('create_time', '')
        desc = item.get('description', '')
        desc_short = desc[:30] + '...' if len(desc) > 30 else desc
        print(f"{idx:2}. ID: {kb_id:3}  |  名称: {name}")
        print(f"   创建者: {user_name}  |  创建时间: {create_time}")
        print(f"   描述: {desc_short}")
        print("-" * 80)

    return all_items


# ---------- 语义分析函数（与之前相同） ----------
def analyze_chunk_depth(chunk_text, llm):
    if not chunk_text or not chunk_text.strip():
        return {"score": 0, "reason": "空文本"}
    truncated = chunk_text[:2000] + ("..." if len(chunk_text) > 2000 else "")
    prompt = f"""请分析以下文本片段的语义深度，从信息丰富度、层次结构（是否有多个逻辑层次）、上下文完整性等角度评估，给出1-10的评分（10为最深），并简要说明理由。

只返回纯JSON格式，例如：{{"score": 8, "reason": "该片段包含了定义、举例和对比，层次清晰，信息密度高"}}

文本片段：
{truncated}
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0]
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0]
        data = json.loads(content)
        return {"score": data.get("score", 0), "reason": data.get("reason", "")}
    except Exception as e:
        print(f"⚠️ 深度分析出错: {e}")
        return {"score": 0, "reason": f"分析失败: {str(e)}"}


def analyze_chunk_overlap(text1, text2, llm):
    if not text1.strip() or not text2.strip():
        return {"overlap_score": 0, "reason": "存在空文本"}
    t1 = text1[:1000] + ("..." if len(text1) > 1000 else "")
    t2 = text2[:1000] + ("..." if len(text2) > 1000 else "")
    prompt = f"""请评估以下两个连续文本片段之间的语义重合度，即内容重叠程度，0表示完全不同，10表示几乎完全相同。同时判断这种重合是否合理（如果切分导致过度重复则不合理，如果保持了连贯性则合理）。

返回JSON格式：{{"overlap_score": 0-10, "reason": "说明"}}

片段1：
{t1}

片段2：
{t2}
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0]
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0]
        data = json.loads(content)
        return {"overlap_score": data.get("overlap_score", 0), "reason": data.get("reason", "")}
    except Exception as e:
        print(f"⚠️ 重合度分析出错: {e}")
        return {"overlap_score": 0, "reason": f"分析失败: {str(e)}"}


def analyze_chunks(chunks, llm, max_depth=MAX_ANALYZE_CHUNKS, max_pairs=MAX_ANALYZE_PAIRS):
    if not chunks:
        return {"depth": [], "overlap": [], "error": "无chunks"}
    print("\n🧠 开始语义分析（使用大模型）...")
    depth_results = []
    valid_chunks = [c for c in chunks if c.get('text', '').strip()]
    analyze_list = valid_chunks[:max_depth]
    for i, chunk in enumerate(analyze_list):
        text = chunk.get('text', '')
        print(f"  → 分析第 {i+1} 个块的语义深度...")
        result = analyze_chunk_depth(text, llm)
        depth_results.append({
            "chunk_index": i,
            "score": result.get("score"),
            "reason": result.get("reason")
        })
    overlap_results = []
    for i in range(len(valid_chunks) - 1):
        if i >= max_pairs:
            break
        text1 = valid_chunks[i].get('text', '')
        text2 = valid_chunks[i+1].get('text', '')
        print(f"  → 分析第 {i+1}-{i+2} 块的重合度...")
        result = analyze_chunk_overlap(text1, text2, llm)
        overlap_results.append({
            "pair": (i, i+1),
            "overlap_score": result.get("overlap_score"),
            "reason": result.get("reason")
        })
    return {"depth": depth_results, "overlap": overlap_results}


def print_analysis_report(analysis_result):
    if not analysis_result or analysis_result.get("error"):
        print("❌ 无分析结果或出错")
        return
    print("\n" + "=" * 60)
    print("📊 语义分析报告")
    print("=" * 60)
    depth = analysis_result.get("depth", [])
    if depth:
        print("\n【段内语义深度】")
        for item in depth:
            idx = item.get("chunk_index", "?")
            score = item.get("score", "N/A")
            reason = item.get("reason", "")
            print(f"  Chunk {idx}: 评分 {score}/10")
            print(f"    理由: {reason}")
    overlap = analysis_result.get("overlap", [])
    if overlap:
        print("\n【段间语义重合度】")
        for item in overlap:
            pair = item.get("pair", ("?", "?"))
            score = item.get("overlap_score", "N/A")
            reason = item.get("reason", "")
            print(f"  块 {pair[0]} ↔ 块 {pair[1]}: 重合度 {score}/10")
            print(f"    理由: {reason}")
    print("=" * 60)


# ---------- 主流程 ----------
def main():
    # ===== 第一步：交互式配置切分参数 =====
    chunk_size, chunk_overlap, separator, separator_rule = interactive_config(FILE_PATH)

    # ===== 第二步：选择知识库来源（新建或复用） =====
    print("\n" + "=" * 50)
    source_choice = input("请选择知识库来源: 1. 新建  |  2. 输入已有 knowledge_id: ").strip()
    knowledge_id = None

    if source_choice == "2":
        # --- 复用模式：先列出所有知识库，再让用户选择 ---
        print("\n正在获取知识库列表...")
        kb_list = list_knowledge_bases()  # 打印并返回列表
        if not kb_list:
            print("❌ 没有可用的知识库，请先创建或检查网络。")
            return
        # 让用户输入 ID
        while True:
            kid_input = input("请输入想要复用的知识库 ID（输入数字 ID）: ").strip()
            if not kid_input.isdigit():
                print("❌ 请输入有效的数字 ID")
                continue
            knowledge_id = int(kid_input)
            # 验证该 ID 是否在列表中
            ids = [item.get('id') for item in kb_list]
            if knowledge_id in ids:
                break
            else:
                print(f"❌ 知识库 ID {knowledge_id} 不在列表中，请重新输入。")
        print(f"✅ 将直接复用已有知识库 ID: {knowledge_id}（跳过新建步骤）")
    else:
        # --- 新建模式 ---
        print("\n【步骤 1/5】创建知识库")
        knowledge_id = create_knowledge_base(KB_NAME, KB_DESCRIPTION, MODEL_ID)
        if not knowledge_id:
            print("❌ 流程终止: 知识库创建失败")
            return

    # ===== 第三步：执行后续流程（上传、预览、分析、处理） =====
    # 为了计数清晰，调整步骤编号。如果是复用的，步骤1就不存在了，但我们依然按序号执行
    step_counter = 1
    if source_choice != "2":
        step_counter = 2  # 如果已经新建了，下一步就是步骤2

    print("\n" + "=" * 50)
    print("🚀 继续执行后续流程: 上传文件 → 预览切分 → 语义分析 → 正式处理")
    print("=" * 50)

    # 步骤：上传文件
    print(f"\n【步骤 {step_counter}/4】上传文件")
    presigned_url = upload_file(knowledge_id, FILE_PATH)
    if not presigned_url:
        print("❌ 流程终止: 文件上传失败")
        return
    step_counter += 1

    # 步骤：预览切分
    print(f"\n【步骤 {step_counter}/4】预览文档切分效果")
    preview_id = preview_file(
        knowledge_id, presigned_url,
        separator, separator_rule,
        chunk_size, chunk_overlap
    )
    preview_result = None
    if preview_id:
        preview_result = get_preview_result(preview_id)
        if preview_result:
            print_preview_result(preview_result)
        else:
            print("⚠️ 获取预览结果失败，将跳过分析步骤")
    else:
        print("⚠️ 提交预览失败，将跳过分析步骤")
    step_counter += 1

    # 步骤：语义分析
    if preview_result and preview_result.get('chunks'):
        print(f"\n【步骤 {step_counter}/4】语义分析（利用大模型）")
        llm = create_llm()
        analysis = analyze_chunks(preview_result['chunks'], llm)
        print_analysis_report(analysis)
    else:
        print(f"\n【步骤 {step_counter}/4】跳过语义分析（无预览结果或 chunks 为空）")
    step_counter += 1

    # 步骤：正式处理
    print(f"\n【步骤 {step_counter}/4】提交正式处理任务")
    file_name = os.path.basename(FILE_PATH)
    process_file(knowledge_id, file_name, presigned_url,
                 separator, separator_rule, chunk_size, chunk_overlap)

    print("\n" + "=" * 50)
    print("✅ 完整流程执行完毕！")
    print("=" * 50)


if __name__ == "__main__":
    main()