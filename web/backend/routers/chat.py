"""
DeepSeek AI 聊天路由
===================
提供 SSE 流式聊天端点，基于 DeepSeek API 实现碳汇智能问答
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from pathlib import Path
import json
import asyncio

router = APIRouter(prefix="/api/chat", tags=["AI 助手"])

# 凭据路径
CREDENTIALS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "credentials" / "deepseek.json"
WEB_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ============ 数据模型 ============

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    region: Optional[str] = None  # 当前选中的研究区
    year: Optional[int] = None    # 当前选中的年份


# ============ 辅助函数 ============

def load_credentials() -> dict:
    """加载 DeepSeek API 凭据"""
    if not CREDENTIALS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="DeepSeek API 凭据未配置，请在 config/credentials/deepseek.json 中设置 api_key"
        )
    with open(CREDENTIALS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_system_prompt(region: Optional[str] = None, year: Optional[int] = None) -> str:
    """构建系统提示词，注入项目数据上下文"""

    region_name = ""
    if region == "ninger":
        region_name = "宁洱哈尼族彝族自治县"
    elif region == "shuangbai":
        region_name = "双柏县"

    # 加载统计数据作为上下文
    stats_context = ""
    try:
        stats_path = WEB_DATA_DIR / "stats" / "all_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                all_stats = json.load(f)
            stats_context = "\n\n## 当前统计数据\n```json\n" + json.dumps(all_stats, ensure_ascii=False, indent=2)[:3000] + "\n```"
    except Exception:
        pass

    # 加载变化检测数据
    changes_context = ""
    try:
        changes_path = WEB_DATA_DIR / "stats" / "all_changes.json"
        if changes_path.exists():
            with open(changes_path, 'r', encoding='utf-8') as f:
                all_changes = json.load(f)
            changes_context = "\n\n## 碳汇变化检测数据\n```json\n" + json.dumps(all_changes, ensure_ascii=False, indent=2)[:2000] + "\n```"
    except Exception:
        pass

    # 加载模型指标
    model_context = ""
    for r in ["ninger", "shuangbai"]:
        try:
            metrics_path = WEB_DATA_DIR / "stats" / f"model_metrics_{r}.json"
            if metrics_path.exists():
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                rname = "宁洱县" if r == "ninger" else "双柏县"
                model_context += f"\n### {rname}模型指标\n"
                model_context += f"- 模型类型: {metrics.get('model_type', 'N/A')}\n"
                model_context += f"- 交叉验证OA: {metrics.get('cv_oa_mean', 0)*100:.1f}%\n"
                model_context += f"- 验证集OA: {metrics.get('val_oa', 0)*100:.1f}%\n"
                model_context += f"- Kappa系数: {metrics.get('val_kappa', 0):.3f}\n"
                model_context += f"- 特征数量: {metrics.get('n_features', 'N/A')}\n"
                if metrics.get('best_params'):
                    model_context += f"- 最佳参数: n_estimators={metrics['best_params'].get('n_estimators')}, max_depth={metrics['best_params'].get('max_depth')}\n"
        except Exception:
            pass

    current_context = ""
    if region_name and year:
        current_context = f"\n\n## 用户当前视角\n用户正在查看 **{region_name}** 的 **{year}年** 碳汇数据。"

    prompt = f"""你是"树智碳汇"平台的AI智能助手，专注于森林碳汇领域的知识问答和数据分析。

## 项目背景
树智碳汇（TreeWit Carbon）是一个基于多源遥感数据与随机森林机器学习的森林碳汇时空动态监测平台。
- 研究区：云南省宁洱哈尼族彝族自治县（普洱市）和双柏县（楚雄州）
- 时间范围：2019年基准训练+2023-2025年碳汇变化追踪
- 数据来源：Landsat-8/9卫星影像、SRTM DEM地形数据、PKU-AGB（北京大学森林地上生物量密度数据集）
- 特征空间：28维特征（7个Landsat光谱波段 + 13个植被指数 + 8个地形因子）
- 分类器：HierarchicalClassifier分层二分类（L1 vs L23, L2 vs L3）
- 分级体系：低碳密度(Low)、中碳密度(Medium)、高碳密度(High) 三级

## 特征说明
- B1(沿海气溶胶)、B2(蓝光)、B3(绿光)、B4(红光)、B5(近红外)、B6(SWIR1短波红外)、B7(SWIR2短波红外)
- NDVI(归一化植被指数)、EVI(增强植被指数)、NDMI(归一化水分指数)、SAVI(土壤调节植被指数)、NDWI(归一化水体指数)
- MSAVI(改进土壤调节植被指数)、NBR(归一化燃烧比)、SR_B5B4/SR_B6B5/SR_B7B5(简单波段比值)、TCB/TCG/TCW(缨帽变换亮度/绿度/湿度)
- elevation(海拔)、slope(坡度)、aspect(坡向)、twi(地形湿度指数)、pcurv(剖面曲率)、ccurv(平面曲率)、slope_pos(坡位)、roughness(粗糙度)

## 你的能力
1. 解答碳汇、碳交易、遥感监测相关专业知识
2. 分析平台数据，解读统计指标和趋势
3. 解释模型原理、特征含义、不确定性来源
4. 提供林业碳汇政策解读和应用建议
5. 根据用户当前查看的研究区和年份提供针对性分析

## 回答要求
- 使用中文回答
- 数据相关回答要基于下方提供的真实数据，不要编造数字
- 适当使用表格、列表等结构化格式
- 如果涉及专业知识，给出简洁准确的解释
- 如果数据不足以回答，诚实说明

{stats_context}{changes_context}{model_context}{current_context}"""

    return prompt


# ============ API 路由 ============

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式聊天端点"""

    creds = load_credentials()

    client = OpenAI(
        api_key=creds["api_key"],
        base_url=creds.get("base_url", "https://api.deepseek.com")
    )

    model = creds.get("model", "deepseek-chat")

    # 构建消息列表
    system_prompt = build_system_prompt(region=request.region, year=request.year)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    async def event_generator():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=2048,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    # SSE 格式: data: {json}\n\n
                    data = json.dumps({"content": delta.content}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0)  # 让出事件循环

            # 发送结束标记
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/status")
async def chat_status():
    """检查 AI 助手配置状态"""
    configured = CREDENTIALS_PATH.exists()
    model = ""
    if configured:
        try:
            creds = load_credentials()
            model = creds.get("model", "deepseek-chat")
        except Exception:
            configured = False
    return {
        "configured": configured,
        "model": model,
        "service": "DeepSeek"
    }
