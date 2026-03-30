from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import streamlit as st
import yaml

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore


# ----------------------------
# Constants / Defaults
# ----------------------------

APP_TITLE = "FDA 510(k) Agentic Review System — WOW Edition"
APP_ICON = "🧾"

SUPPORTED_UI_LANGS = {"en": "English", "zh-TW": "繁體中文"}
SUPPORTED_OUTPUT_LANGS = {"en": "English", "zh-TW": "繁體中文"}

PROVIDERS = ["gemini", "openai", "anthropic", "grok"]

MODEL_CATALOG: Dict[str, List[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview"],
    "anthropic": [
        # Keep as a curated baseline; can be expanded via agents.yaml or environment.
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ],
    "grok": ["grok-4-fast-reasoning", "grok-3-mini"],
}

DEFAULT_MAX_TOKENS = 12000
DEFAULT_TEMPERATURE = 0.2

# Keyword highlight default color (coral)
DEFAULT_CORAL = "#FF7F50"

# Default 510(k) review guidance (Traditional Chinese) — provided by user (kept as-is).
DEFAULT_GUIDANCE_ZH_TW = r"""
診斷用超音波影像系統暨超音波轉換器(探頭) 510(k) 審查與臨床前測試指引手冊
第一章：前言與指引說明
本指引手冊係依據衛生福利部於 107 年 1 月 16 日公布之「診斷用超音波影像系統暨超音波轉換器(探頭)」臨床前測試基準（以下簡稱本檢測基準），並結合 GE Healthcare 針對 Vscan Air™ 系統之 510(k) 申請實務案例撰寫而成。本文件旨在為醫療器材製造商在辦理產品查驗登記或進行 510(k) 實質等同性（Substantial Equivalence）比對時，提供詳盡的臨床前測試資料檢附建議。

1.1 適用目的與範圍
本指引主要提供醫療器材廠商在準備臨床前測試資料時的具體項目建議。這些資料是確保產品在安全性與效能上達到法規要求的關鍵。本基準適用於人體組織監測及診斷之超音波器材，包含：
- 超音波脈動都卜勒式影像系統 (Ultrasonic pulsed doppler imaging system)
- 超音波回音影像系統 (Ultrasonic Pulsed Echo Imaging System)
- 診斷用超音波轉換器（探頭） (Diagnostic ultrasonic transducer)
需注意，本基準並不適用於「超音波治療設備」。所有申請案仍應符合相關法規，並依產品之特定結構、材質與宣稱效能進行完整的驗證評估。

1.2 測試資料基本要求
製造商應檢附之臨床前測試資料必須包含：
- 檢驗規格：明列各測試項目之合格範圍（Acceptance Criteria）及其制定依據。
- 方法：詳細描述測試流程。
- 原始檢驗紀錄：提供測試過程中的數據紀錄。
- 檢驗成績書：總結測試結果並判定是否符合規格。
若產品未進行指引表列之項目，製造商須提供科學性評估報告或相關文獻，證明產品安全性與功能性不受影響。

（下略：此處可保留完整指引；若你希望完整嵌入，請將全文貼入本常數。）
""".strip()

DEFAULT_510K_REPORT_PROMPT = r"""
**STEP 1: ANALYZE THE USER'S DOCUMENT**
First, carefully read and analyze the user's 510(k) document to understand:
- The type of medical device being reviewed
- Key technical specifications and features
- Safety and performance requirements
- Regulatory standards referenced
- Clinical applications and intended use

**STEP 2: PROCESS THE REVIEW TEMPLATE**
Examine the review template structure to understand:
- Chapter organization and flow
- Required sections and subsections
- Table formats and entity explanations
- Testing requirements and standards

**STEP 3: CREATE COMPREHENSIVE REVIEW INSTRUCTIONS**
Using the template structure and the user's document content, create a detailed 510(k) review instruction document in markdown format that includes:

1. **Structured chapters** covering all relevant aspects:
   - Introduction and scope
   - Device classification
   - Product specifications
   - Safety and performance testing
   - Biocompatibility requirements
   - Substantial equivalence analysis
   - Key entity explanations

2. **Comprehensive review checklist** organized by category of submission materials:
   - Administrative documents
   - Technical specifications
   - Safety testing reports
   - Performance validation data
   - Clinical information
   - Labeling and instructions for use
   - Manufacturing information
   - Sterilization/disinfection data (if applicable)

Each checklist item should include:
- Item description
- Required documentation
- Applicable standards
- Acceptance criteria
- Review notes/considerations

**STEP 4: ADD 5 ADDITIONAL TABLES**
Create 5 comprehensive tables with context based on the review instructions. These tables should cover different aspects such as:
- Comparative analysis tables
- Testing requirements matrices
- Risk assessment tables
- Standards compliance tables
- Timeline or process flow tables

Each table should have clear headers, well-organized rows, and contextual explanations.

**STEP 5: ADD 20 ENTITIES WITH CONTEXT**
Provide detailed explanations for 20 key entities/terms that appear in the review instructions. For each entity, include:
- Clear definition
- Regulatory context
- Relevance to the specific device
- Related standards or requirements
- Practical implications for reviewers

**STRICT RULES**
- Use Markdown headings and tables.
- Do NOT invent testing outcomes, clearances, or standards compliance. If something is missing, explicitly write “Not provided”.
- Include a short “Provenance & Assumptions” section stating what inputs were provided (510(k) doc type, guidance source) and what may be missing.
- Target length: 2000–3000 words.
""".strip()

# Minimal SKILL.md fallback if a file is not present.
FALLBACK_SKILL_MD = r"""
You are an expert FDA 510(k) reviewer.
Core principles:
- Evidence discipline: never invent data; mark missing items as “Not provided”.
- Output must be clean Markdown with headings, lists, and tables when appropriate.
- Provide actionable reviewer notes and acceptance criteria framing.
- Be clear, concise, and professional.

Formatting:
- Use ## for sections, ### for subsections.
- Use tables for matrices and checklists.
- Highlight key keywords using **bold** (renderer styles **bold** as coral in UI).
""".strip()


# ----------------------------
# WOW Painter Styles (20)
# ----------------------------

@dataclass(frozen=True)
class PainterStyle:
    key: str
    name: str
    accent: str
    accent2: str
    bg_tint_light: str
    bg_tint_dark: str
    card_tint_light: str
    card_tint_dark: str


PAINTER_STYLES: List[PainterStyle] = [
    PainterStyle("davinci", "Leonardo da Vinci", "#8B5E3C", "#C9A26A", "#FBF1E1", "#1C1612", "#FFF7EC", "#241C17"),
    PainterStyle("vangogh", "Vincent van Gogh", "#1E4ED8", "#F59E0B", "#F2F6FF", "#0B1220", "#FFFFFF", "#101A2E"),
    PainterStyle("monet", "Claude Monet", "#60A5FA", "#FBCFE8", "#F7FBFF", "#0E1A22", "#FFFFFF", "#12212B"),
    PainterStyle("picasso", "Pablo Picasso", "#111827", "#F37021", "#FFF7ED", "#0B0F14", "#FFFFFF", "#121826"),
    PainterStyle("dali", "Salvador Dalí", "#7C3AED", "#F59E0B", "#FBF7FF", "#140B1F", "#FFFFFF", "#1B102A"),
    PainterStyle("rembrandt", "Rembrandt", "#92400E", "#F59E0B", "#FEF3C7", "#0F0A06", "#FFF7ED", "#1A120C"),
    PainterStyle("kahlo", "Frida Kahlo", "#DC2626", "#16A34A", "#FFF5F5", "#140C0C", "#FFFFFF", "#1F1212"),
    PainterStyle("hokusai", "Katsushika Hokusai", "#2563EB", "#0EA5E9", "#F0F9FF", "#06121F", "#FFFFFF", "#0C1B2B"),
    PainterStyle("pollock", "Jackson Pollock", "#111827", "#A3A3A3", "#FAFAFA", "#0A0A0A", "#FFFFFF", "#121212"),
    PainterStyle("klimt", "Gustav Klimt", "#B45309", "#FDE047", "#FFFBEB", "#161106", "#FFFFFF", "#1E1709"),
    PainterStyle("okeeffe", "Georgia O’Keeffe", "#DB2777", "#F59E0B", "#FFF1F2", "#170B10", "#FFFFFF", "#211018"),
    PainterStyle("warhol", "Andy Warhol", "#EC4899", "#22C55E", "#FFF7FB", "#0F0A10", "#FFFFFF", "#1A121B"),
    PainterStyle("hopper", "Edward Hopper", "#0F766E", "#F59E0B", "#F0FDFA", "#071917", "#FFFFFF", "#0B2320"),
    PainterStyle("caravaggio", "Caravaggio", "#F59E0B", "#7C2D12", "#FFF7ED", "#050505", "#FFFFFF", "#101010"),
    PainterStyle("basquiat", "Jean-Michel Basquiat", "#F97316", "#111827", "#FFF7ED", "#0B0F14", "#FFFFFF", "#131A24"),
    PainterStyle("vermeer", "Johannes Vermeer", "#2563EB", "#FDE68A", "#F5FAFF", "#0A1422", "#FFFFFF", "#101C2E"),
    PainterStyle("kandinsky", "Wassily Kandinsky", "#EF4444", "#3B82F6", "#FFF5F5", "#0B1020", "#FFFFFF", "#121A2E"),
    PainterStyle("escher", "M.C. Escher", "#6B7280", "#111827", "#F9FAFB", "#070A0F", "#FFFFFF", "#121826"),
    PainterStyle("kusama", "Yayoi Kusama", "#DC2626", "#111827", "#FFF5F5", "#0A0A0A", "#FFFFFF", "#121212"),
    PainterStyle("cezanne", "Paul Cézanne", "#78350F", "#A16207", "#FFFBF5", "#120D08", "#FFFFFF", "#1E150D"),
]


# ----------------------------
# Session State Initialization
# ----------------------------

def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("ui_lang", "en")
    ss.setdefault("theme_mode", "light")
    ss.setdefault("painter_style_key", "picasso")

    ss.setdefault("keys", {})  # provider -> key (user-supplied only)
    ss.setdefault("key_source", {})  # provider -> 'env'|'user'|'none'

    ss.setdefault("telemetry", {
        "provider_last_latency_ms": {},
        "provider_last_error": {},
        "total_calls": 0,
        "total_est_tokens_in": 0,
        "total_est_tokens_out": 0,
        "run_events": [],  # timeline events
    })

    ss.setdefault("docs", {
        "submission_docs": [],  # list of NormalizedDoc dicts
        "guidance_doc": None,   # NormalizedDoc dict
    })

    ss.setdefault("workspace", {
        "name": f"workspace-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "created_at": dt.datetime.now().isoformat(),
    })

    ss.setdefault("report", {
        "prompt": DEFAULT_510K_REPORT_PROMPT,
        "output_lang": "en",
        "provider": "gemini",
        "model": MODEL_CATALOG["gemini"][0],
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "markdown": "",
        "last_run_meta": None,
    })

    ss.setdefault("note_keeper", {
        "input_text": "",
        "output_text": "",
        "provider": "gemini",
        "model": MODEL_CATALOG["gemini"][0],
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "keywords_color": DEFAULT_CORAL,
        "keywords_list": "",
        "last_magic": None,
    })

    ss.setdefault("agents_yaml", {
        "raw": "",
        "standardized": "",
        "transform_report": "",
        "agents": [],  # list of standardized agents dicts
    })

    ss.setdefault("runner", {
        "n_agents": 3,
        "input_text": "",
        "use_report_as_input": True,
        "configs": [],   # per-agent runtime configs
        "outputs": [],   # per-agent outputs (editable)
        "logs": [],      # run logs
    })

    ss.setdefault("skill_md", "")
    ss.setdefault("default_guidance_enabled", True)


# ----------------------------
# Utility: UI Text (EN/zh-TW)
# ----------------------------

TEXT = {
    "en": {
        "sidebar_title": "Controls",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark",
        "ui_language": "UI language",
        "painter_style": "Painter style",
        "jackpot": "Jackpot (random style)",
        "api_keys": "API Keys",
        "key_loaded_env": "Loaded from environment",
        "key_missing": "Missing — enter key (session-only)",
        "clear_session_keys": "Clear session keys",
        "tabs": ["Dashboard", "510(k) Report", "Prompt Runner", "AI Note Keeper", "Agents YAML", "Document Review"],
        "warning_privacy": "Privacy: Do not submit PHI or confidential data unless you have appropriate agreements and provider settings.",
    },
    "zh-TW": {
        "sidebar_title": "控制面板",
        "theme": "主題",
        "light": "亮色",
        "dark": "暗色",
        "ui_language": "介面語言",
        "painter_style": "畫家風格",
        "jackpot": "幸運轉盤（隨機風格）",
        "api_keys": "API 金鑰",
        "key_loaded_env": "已由環境載入",
        "key_missing": "缺少—請輸入金鑰（僅本次工作階段）",
        "clear_session_keys": "清除工作階段金鑰",
        "tabs": ["儀表板", "510(k) 報告", "代理執行器", "AI 筆記整理", "Agents YAML", "文件審閱"],
        "warning_privacy": "隱私提醒：除非你已具備適當合約與供應商設定，否則請勿提交 PHI 或機密資料。",
    }
}


def t(key: str) -> str:
    lang = st.session_state.get("ui_lang", "en")
    return TEXT.get(lang, TEXT["en"]).get(key, key)


# ----------------------------
# Utility: CSS (Theme + Painter)
# ----------------------------

def _get_style(style_key: str) -> PainterStyle:
    for s in PAINTER_STYLES:
        if s.key == style_key:
            return s
    return PAINTER_STYLES[0]


def inject_css() -> None:
    ss = st.session_state
    theme = ss["theme_mode"]
    ps = _get_style(ss["painter_style_key"])

    # Variables tuned for both light and dark.
    if theme == "light":
        bg = ps.bg_tint_light
        card = ps.card_tint_light
        text = "#1F2937"
        muted = "#6B7280"
        border = "rgba(17,24,39,0.10)"
    else:
        bg = ps.bg_tint_dark
        card = ps.card_tint_dark
        text = "#E5E7EB"
        muted = "#9CA3AF"
        border = "rgba(229,231,235,0.12)"

    css = f"""
    <style>
      :root {{
        --wow-bg: {bg};
        --wow-card: {card};
        --wow-text: {text};
        --wow-muted: {muted};
        --wow-border: {border};
        --wow-accent: {ps.accent};
        --wow-accent2: {ps.accent2};
        --wow-coral: {DEFAULT_CORAL};
      }}

      .stApp {{
        background: var(--wow-bg);
        color: var(--wow-text);
      }}

      /* Card-like containers */
      div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: var(--wow-border) !important;
      }}

      /* Buttons */
      .stButton>button {{
        border-radius: 12px;
        border: 1px solid var(--wow-border);
        background: linear-gradient(135deg, var(--wow-accent), var(--wow-accent2));
        color: white;
        font-weight: 700;
      }}
      .stButton>button:hover {{
        filter: brightness(1.03);
        transform: translateY(-1px);
      }}

      /* Inputs */
      .stTextInput input, .stTextArea textarea, .stSelectbox div, .stNumberInput input {{
        border-radius: 12px !important;
      }}

      /* Keyword highlight in Markdown: render **bold** as coral */
      .wow-markdown strong {{
        color: var(--wow-coral);
      }}

      /* Status pills */
      .wow-pill {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 12px;
        border: 1px solid var(--wow-border);
        background: rgba(255,255,255,0.18);
      }}
      .wow-pill.ok {{ background: rgba(34,197,94,0.18); }}
      .wow-pill.warn {{ background: rgba(245,158,11,0.18); }}
      .wow-pill.err {{ background: rgba(239,68,68,0.18); }}

      /* Nice separators */
      hr {{
        border: none;
        height: 1px;
        background: var(--wow-border);
        margin: 10px 0;
      }}

      /* Streamlit markdown containers */
      .wow-panel {{
        border: 1px solid var(--wow-border);
        background: var(--wow-card);
        border-radius: 16px;
        padding: 14px;
      }}

      /* Small “WOW” sparkle */
      .wow-spark {{
        background: radial-gradient(circle at 20% 20%, var(--wow-accent2), transparent 55%),
                    radial-gradient(circle at 80% 30%, var(--wow-accent), transparent 55%);
        border-radius: 16px;
        border: 1px solid var(--wow-border);
        padding: 12px 14px;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ----------------------------
# Utility: Telemetry
# ----------------------------

def log_event(kind: str, detail: Dict[str, Any]) -> None:
    tel = st.session_state["telemetry"]
    tel["run_events"].append({
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "detail": detail,
    })


def set_provider_latency(provider: str, ms: int) -> None:
    st.session_state["telemetry"]["provider_last_latency_ms"][provider] = ms


def set_provider_error(provider: str, err: str) -> None:
    st.session_state["telemetry"]["provider_last_error"][provider] = err


def est_tokens(text: str) -> int:
    # Rough estimate: 1 token ~ 4 chars (English). Works as a pragmatic heuristic.
    if not text:
        return 0
    return max(1, len(text) // 4)


# ----------------------------
# API Keys: Environment + User Input
# ----------------------------

ENV_KEY_MAP = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "grok": ["GROK_API_KEY", "XAI_API_KEY"],
}


def get_env_key(provider: str) -> Optional[str]:
    for k in ENV_KEY_MAP.get(provider, []):
        v = os.environ.get(k)
        if v:
            return v.strip()
    return None


def get_api_key(provider: str) -> Tuple[Optional[str], str]:
    """
    Returns (key, source) where source is 'env'|'user'|'none'.
    """
    env = get_env_key(provider)
    if env:
        st.session_state["key_source"][provider] = "env"
        return env, "env"

    user = st.session_state["keys"].get(provider)
    if user:
        st.session_state["key_source"][provider] = "user"
        return user, "user"

    st.session_state["key_source"][provider] = "none"
    return None, "none"


def sidebar_api_key_controls() -> None:
    st.subheader(t("api_keys"))
    st.caption(t("warning_privacy"))

    for provider in PROVIDERS:
        key, source = get_api_key(provider)
        cols = st.columns([1.2, 1.8])
        with cols[0]:
            st.write(provider.upper())
        with cols[1]:
            if source == "env":
                st.markdown(f"<span class='wow-pill ok'>{t('key_loaded_env')}</span>", unsafe_allow_html=True)
            else:
                # Only show input when not available via environment.
                st.markdown(f"<span class='wow-pill warn'>{t('key_missing')}</span>", unsafe_allow_html=True)
                user_in = st.text_input(
                    label=f"{provider.upper()} key",
                    type="password",
                    key=f"key_input_{provider}",
                    placeholder="••••••••••••••••",
                    label_visibility="collapsed",
                )
                if user_in:
                    st.session_state["keys"][provider] = user_in.strip()
                    st.session_state["key_source"][provider] = "user"

    if st.button(t("clear_session_keys"), use_container_width=True):
        st.session_state["keys"] = {}
        for provider in PROVIDERS:
            if st.session_state["key_source"].get(provider) == "user":
                st.session_state["key_source"][provider] = "none"
        log_event("security", {"action": "clear_session_keys"})


# ----------------------------
# Providers: REST Calls
# ----------------------------

def llm_generate(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (text, meta) where meta includes provider/model/latency/usage-estimates/errors.
    Uses REST endpoints to avoid heavyweight SDK dependencies.
    """
    api_key, source = get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"Missing API key for provider: {provider}")

    start = time.time()
    meta: Dict[str, Any] = {"provider": provider, "model": model, "key_source": source}
    text_out = ""

    try:
        if provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
            text_out = data["choices"][0]["message"]["content"]
            meta["raw"] = {"id": data.get("id"), "usage": data.get("usage")}

        elif provider == "grok":
            # xAI offers an OpenAI-compatible endpoint.
            url = "https://api.x.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
            text_out = data["choices"][0]["message"]["content"]
            meta["raw"] = {"id": data.get("id"), "usage": data.get("usage")}

        elif provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            r = requests.post(url, headers=headers, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
            # Anthropics messages content is a list; extract text blocks.
            blocks = data.get("content", [])
            parts = []
            for b in blocks:
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
            text_out = "\n".join(parts).strip()
            meta["raw"] = {"id": data.get("id"), "usage": data.get("usage")}

        elif provider == "gemini":
            # Google Generative Language API (public endpoint) — systemInstruction supported in v1beta.
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }
            r = requests.post(url, headers=headers, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"Gemini returned no candidates: {json.dumps(data)[:2000]}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text_out = "\n".join([p.get("text", "") for p in parts]).strip()
            meta["raw"] = {"usageMetadata": data.get("usageMetadata")}

        else:
            raise RuntimeError(f"Unsupported provider: {provider}")

    except Exception as e:
        set_provider_error(provider, str(e))
        raise
    finally:
        elapsed_ms = int((time.time() - start) * 1000)
        set_provider_latency(provider, elapsed_ms)
        meta["latency_ms"] = elapsed_ms

    # Heuristic usage estimates if provider doesn't return usage
    meta["est_tokens_in"] = est_tokens(system_prompt) + est_tokens(user_prompt)
    meta["est_tokens_out"] = est_tokens(text_out)

    tel = st.session_state["telemetry"]
    tel["total_calls"] += 1
    tel["total_est_tokens_in"] += meta["est_tokens_in"]
    tel["total_est_tokens_out"] += meta["est_tokens_out"]

    return text_out, meta


# ----------------------------
# Documents: Normalization & Extraction
# ----------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def detect_lang_heuristic(text: str) -> str:
    if not text:
        return "unknown"
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if zh > en * 2:
        return "zh"
    if en > zh * 2:
        return "en"
    return "mixed"


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    if PdfReader is None:
        return "", {"method": "pdf_text", "warnings": ["PyPDF not installed; cannot extract PDF text."]}

    warnings: List[str] = []
    page_map: Dict[int, str] = {}
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        texts: List[str] = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_map[i + 1] = page_text
            texts.append(page_text)
        full = "\n\n".join(texts).strip()
        if len(full) < 50:
            warnings.append("PDF text extraction yielded very little text; scanned PDF may require OCR.")
        return full, {"method": "pdf_text", "warnings": warnings, "page_map": page_map}
    except Exception as e:
        return "", {"method": "pdf_text", "warnings": [f"PDF extraction error: {e}"]}


def normalize_uploaded_file(uploaded) -> Dict[str, Any]:
    file_bytes = uploaded.read()
    filename = uploaded.name
    mime = uploaded.type or "application/octet-stream"
    ext = filename.lower().split(".")[-1] if "." in filename else ""

    extracted = ""
    meta: Dict[str, Any] = {"extraction_method": "text", "extraction_warnings": []}

    if ext in ("txt", "md"):
        try:
            extracted = file_bytes.decode("utf-8", errors="replace")
            meta["extraction_method"] = "text"
        except Exception as e:
            meta["extraction_warnings"].append(f"Decode error: {e}")
            extracted = file_bytes.decode("utf-8", errors="replace")

    elif ext in ("pdf",):
        extracted, pdf_meta = extract_text_from_pdf(file_bytes)
        meta["extraction_method"] = pdf_meta.get("method", "pdf_text")
        meta["extraction_warnings"] = pdf_meta.get("warnings", [])
        if pdf_meta.get("page_map"):
            meta["page_map"] = pdf_meta["page_map"]

    else:
        meta["extraction_warnings"].append(f"Unsupported extension: .{ext}. Treating as text.")
        extracted = file_bytes.decode("utf-8", errors="replace")

    doc = {
        "doc_id": sha256_bytes(file_bytes)[:16],
        "doc_type": "other",
        "source": "upload",
        "original_filename": filename,
        "mime_type": mime,
        "hash": sha256_bytes(file_bytes),
        "extracted_text": extracted,
        "language_detected": detect_lang_heuristic(extracted),
        **meta,
    }
    return doc


def normalize_pasted_text(text: str, doc_type: str, source: str) -> Dict[str, Any]:
    b = text.encode("utf-8", errors="replace")
    return {
        "doc_id": sha256_bytes(b)[:16],
        "doc_type": doc_type,
        "source": source,
        "original_filename": None,
        "mime_type": "text/plain",
        "hash": sha256_bytes(b),
        "extracted_text": text,
        "language_detected": detect_lang_heuristic(text),
        "extraction_method": "paste",
        "extraction_warnings": [],
    }


def build_skill_prompt() -> str:
    # Try to load SKILL.md content if present (repo file), else fallback.
    if st.session_state.get("skill_md"):
        return st.session_state["skill_md"]

    skill = ""
    try:
        if os.path.exists("SKILL.md"):
            with open("SKILL.md", "r", encoding="utf-8") as f:
                skill = f.read().strip()
    except Exception:
        skill = ""

    if not skill:
        skill = FALLBACK_SKILL_MD

    st.session_state["skill_md"] = skill
    return skill


# ----------------------------
# agents.yaml: Standardization
# ----------------------------

DEFAULT_AGENTS_YAML = """
version: "1.0"
agents:
  - id: "gap_predictor"
    name: "Regulatory Gap Predictor"
    description: "Identify likely missing/deficient sections in the submission."
    provider: "gemini"
    model: "gemini-2.5-flash"
    max_tokens: 12000
    temperature: 0.2
    system_prompt: |
      You are an expert FDA 510(k) reviewer specializing in gap analysis.
      Output Markdown. Never invent evidence; label missing items as Not provided.
    input_contract: "markdown_or_text"
    output_contract: "markdown_with_bullets_and_prioritized_risks"
    ui:
      icon: "alert-triangle"
      tags: ["510k", "gap", "risk"]

  - id: "checklist_builder"
    name: "Submission Checklist Builder"
    description: "Produce a comprehensive checklist with acceptance criteria and reviewer notes."
    provider: "openai"
    model: "gpt-4o-mini"
    max_tokens: 12000
    temperature: 0.2
    system_prompt: |
      You are an FDA 510(k) checklist specialist.
      Output Markdown tables. Never invent evidence; label missing items as Not provided.
    input_contract: "markdown_or_text"
    output_contract: "markdown_tables_checklist"
    ui:
      icon: "check-square"
      tags: ["510k", "checklist"]

  - id: "final_editor"
    name: "Final Regulatory Editor"
    description: "Polish tone, structure, and ensure consistent headings/tables."
    provider: "anthropic"
    model: "claude-3-5-sonnet-latest"
    max_tokens: 12000
    temperature: 0.2
    system_prompt: |
      You are a meticulous regulatory editor.
      Produce clean Markdown. Preserve factual constraints; never invent missing data.
    input_contract: "markdown"
    output_contract: "clean_markdown"
    ui:
      icon: "file-text"
      tags: ["editor", "markdown"]
""".strip()


def _safe_yaml_load(s: str) -> Any:
    return yaml.safe_load(s) if s.strip() else None


def standardize_agents_yaml(raw_yaml: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Accepts YAML content that may be non-standard and returns:
    (standardized_yaml_string, transform_report_markdown, agents_list)
    """
    report_lines: List[str] = []
    obj = None
    try:
        obj = _safe_yaml_load(raw_yaml)
    except Exception as e:
        report_lines.append(f"- ❌ YAML parse error: `{e}`")
        # Return defaults as fallback
        std = DEFAULT_AGENTS_YAML
        agents = _safe_yaml_load(std)["agents"]
        report_lines.append("- ✅ Fell back to bundled default agents.yaml.")
        return std, "\n".join(report_lines), agents

    if obj is None:
        std = DEFAULT_AGENTS_YAML
        agents = _safe_yaml_load(std)["agents"]
        report_lines.append("- ℹ️ Empty YAML provided; using bundled default agents.yaml.")
        return std, "\n".join(report_lines), agents

    # Canonical structure: {version: str, agents: [ ... ]}
    version = str(obj.get("version", "1.0")) if isinstance(obj, dict) else "1.0"
    candidates = None

    if isinstance(obj, dict):
        if "agents" in obj and isinstance(obj["agents"], list):
            candidates = obj["agents"]
        elif "agent" in obj and isinstance(obj["agent"], list):
            candidates = obj["agent"]
            report_lines.append("- 🔁 Mapped key `agent` → `agents`.")
        elif "items" in obj and isinstance(obj["items"], list):
            candidates = obj["items"]
            report_lines.append("- 🔁 Mapped key `items` → `agents`.")
        else:
            # If dict but unknown, treat as single agent dict
            candidates = [obj]
            report_lines.append("- 🔁 YAML root treated as single agent; wrapped into `agents` list.")
    elif isinstance(obj, list):
        candidates = obj
        report_lines.append("- 🔁 YAML root list treated as `agents`.")
    else:
        candidates = []
        report_lines.append("- ⚠️ YAML structure not recognized; produced empty agent list.")

    standardized_agents: List[Dict[str, Any]] = []
    used_ids = set()

    def _infer_provider_from_model(m: str) -> str:
        m0 = (m or "").lower()
        if m0.startswith("gpt-") or "openai" in m0:
            return "openai"
        if m0.startswith("gemini"):
            return "gemini"
        if m0.startswith("claude"):
            return "anthropic"
        if m0.startswith("grok"):
            return "grok"
        return "gemini"

    for idx, a in enumerate(candidates or []):
        if not isinstance(a, dict):
            report_lines.append(f"- ⚠️ Agent at index {idx} is not a dict; skipped.")
            continue

        # Key mapping
        sys_prompt = a.get("system_prompt") or a.get("prompt") or a.get("instruction") or ""
        if "system_prompt" not in a and ("prompt" in a or "instruction" in a):
            report_lines.append(f"- 🔁 Agent[{idx}] mapped prompt/instruction → system_prompt.")

        model = a.get("model") or a.get("llm") or a.get("engine") or ""
        if "model" not in a and ("llm" in a or "engine" in a):
            report_lines.append(f"- 🔁 Agent[{idx}] mapped llm/engine → model.")

        provider = a.get("provider") or a.get("vendor") or ""
        if not provider:
            provider = _infer_provider_from_model(str(model))
            report_lines.append(f"- ℹ️ Agent[{idx}] inferred provider `{provider}` from model `{model}`.")

        max_tokens = a.get("max_tokens") or a.get("maxTokens") or DEFAULT_MAX_TOKENS
        temperature = a.get("temperature", DEFAULT_TEMPERATURE)

        agent_id = str(a.get("id") or a.get("name") or f"agent_{idx}").strip()
        agent_id = re.sub(r"\s+", "_", agent_id.lower())
        if agent_id in used_ids:
            agent_id = f"{agent_id}_{idx}"
            report_lines.append(f"- 🔁 Duplicate id fixed: Agent[{idx}] id → `{agent_id}`.")
        used_ids.add(agent_id)

        name = str(a.get("name") or agent_id).strip()
        desc = str(a.get("description") or "").strip()

        input_contract = str(a.get("input_contract") or "markdown_or_text")
        output_contract = str(a.get("output_contract") or "markdown")

        ui = a.get("ui") if isinstance(a.get("ui"), dict) else {}
        ui.setdefault("icon", "sparkles")
        ui.setdefault("tags", ["custom"])

        # Safety clause enforcement
        safety_clause = "Never invent"
        if sys_prompt and safety_clause.lower() not in sys_prompt.lower():
            sys_prompt = (sys_prompt.strip() + "\n\n" +
                          "Never invent evidence or results. If missing, explicitly write “Not provided”.")
            report_lines.append(f"- ✅ Agent[{idx}] appended safety clause (no fabrication).")

        standardized_agents.append({
            "id": agent_id,
            "name": name,
            "description": desc,
            "provider": provider,
            "model": str(model) if model else MODEL_CATALOG.get(provider, [""])[0],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "system_prompt": sys_prompt.strip() if sys_prompt else "Output Markdown. Never invent evidence; label missing items as Not provided.",
            "input_contract": input_contract,
            "output_contract": output_contract,
            "ui": ui,
        })

    std_obj = {"version": version, "agents": standardized_agents}
    std_yaml = yaml.safe_dump(std_obj, sort_keys=False, allow_unicode=True)

    if not report_lines:
        report_lines.append("- ✅ YAML already compatible with standardized schema (no changes).")

    report_md = "### agents.yaml Transformation Report\n" + "\n".join(report_lines)
    return std_yaml, report_md, standardized_agents


def ensure_agents_loaded() -> None:
    ss = st.session_state["agents_yaml"]
    if not ss["standardized"]:
        # load default into state and standardize (idempotent)
        ss["raw"] = DEFAULT_AGENTS_YAML
        std, rep, agents = standardize_agents_yaml(ss["raw"])
        ss["standardized"] = std
        ss["transform_report"] = rep
        ss["agents"] = agents


# ----------------------------
# Keyword Highlight Helpers
# ----------------------------

def apply_keyword_spans(md_text: str, keywords: List[str], color: str) -> str:
    """
    Wrap matched keywords with <span style="color:...; font-weight:700">keyword</span>.
    This allows user-selected colors (not only coral).
    """
    if not md_text or not keywords:
        return md_text

    # Sort by length desc to avoid partial overlaps.
    kws = [k.strip() for k in keywords if k.strip()]
    kws.sort(key=len, reverse=True)
    if not kws:
        return md_text

    def repl(match):
        s = match.group(0)
        return f"<span style='color:{color}; font-weight:700'>{s}</span>"

    out = md_text
    for k in kws:
        # Word boundary for ASCII words; for CJK allow direct match.
        if re.search(r"[\u4e00-\u9fff]", k):
            pattern = re.escape(k)
        else:
            pattern = r"\b" + re.escape(k) + r"\b"
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def render_markdown(md: str, allow_html: bool = True) -> None:
    st.markdown(f"<div class='wow-markdown'>{md}</div>", unsafe_allow_html=allow_html)


# ----------------------------
# Module: Sidebar Controls
# ----------------------------

def sidebar_controls() -> None:
    st.sidebar.title(t("sidebar_title"))

    # Theme toggle
    theme = st.sidebar.radio(t("theme"), [t("light"), t("dark")], index=0 if st.session_state["theme_mode"] == "light" else 1)
    st.session_state["theme_mode"] = "light" if theme == t("light") else "dark"

    # UI language
    ui_lang_label = st.sidebar.selectbox(
        t("ui_language"),
        options=list(SUPPORTED_UI_LANGS.keys()),
        format_func=lambda k: SUPPORTED_UI_LANGS[k],
        index=list(SUPPORTED_UI_LANGS.keys()).index(st.session_state["ui_lang"]),
    )
    st.session_state["ui_lang"] = ui_lang_label

    # Painter style + Jackpot
    keys = [s.key for s in PAINTER_STYLES]
    current_idx = keys.index(st.session_state["painter_style_key"]) if st.session_state["painter_style_key"] in keys else 0

    chosen = st.sidebar.selectbox(
        t("painter_style"),
        options=keys,
        index=current_idx,
        format_func=lambda k: next((s.name for s in PAINTER_STYLES if s.key == k), k),
    )
    st.session_state["painter_style_key"] = chosen

    if st.sidebar.button(t("jackpot"), use_container_width=True):
        st.session_state["painter_style_key"] = random.choice(keys)
        log_event("ui", {"action": "jackpot_style", "style": st.session_state["painter_style_key"]})

    st.sidebar.divider()
    sidebar_api_key_controls()


# ----------------------------
# Module: Dashboard
# ----------------------------

def dashboard_page() -> None:
    st.header(TEXT[st.session_state["ui_lang"]]["tabs"][0])

    ps = _get_style(st.session_state["painter_style_key"])
    st.markdown(
        f"<div class='wow-spark'><b>WOW UI</b> · {ps.name} · "
        f"{SUPPORTED_UI_LANGS[st.session_state['ui_lang']]} · {st.session_state['theme_mode'].title()}</div>",
        unsafe_allow_html=True
    )

    tel = st.session_state["telemetry"]
    docs = st.session_state["docs"]
    report = st.session_state["report"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total LLM Calls", tel["total_calls"])
    col2.metric("Est. Tokens In", tel["total_est_tokens_in"])
    col3.metric("Est. Tokens Out", tel["total_est_tokens_out"])
    col4.metric("Submission Docs", len(docs["submission_docs"]))

    st.subheader("Provider Health")
    rows = []
    for p in PROVIDERS:
        _, source = get_api_key(p)
        latency = tel["provider_last_latency_ms"].get(p)
        last_err = tel["provider_last_error"].get(p)
        status = "Up" if source in ("env", "user") else "No key"
        if last_err:
            status = "Degraded"
        rows.append({
            "provider": p,
            "status": status,
            "key_source": source,
            "last_latency_ms": latency if latency is not None else "-",
            "last_error": (last_err[:120] + "…") if last_err and len(last_err) > 120 else (last_err or ""),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Workspace Snapshot")
    ws = st.session_state["workspace"]
    st.markdown("<div class='wow-panel'>", unsafe_allow_html=True)
    st.write("Workspace:", ws["name"])
    st.write("Created:", ws["created_at"])
    st.write("Output language (report):", SUPPORTED_OUTPUT_LANGS.get(report["output_lang"], report["output_lang"]))
    st.write("Default report model:", f"{report['provider']} / {report['model']}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Document Ingestion Metrics")
    submission_chars = sum(len(d.get("extracted_text", "") or "") for d in docs["submission_docs"])
    guidance_chars = len((docs["guidance_doc"] or {}).get("extracted_text", "") or "")
    st.write(f"Submission extracted chars: {submission_chars:,}")
    st.write(f"Guidance extracted chars: {guidance_chars:,}")
    if docs["submission_docs"]:
        st.write("Detected languages (submission):", ", ".join(sorted({d.get("language_detected") for d in docs["submission_docs"]})))

    st.subheader("Recent Activity (Session)")
    events = tel["run_events"][-20:]
    if not events:
        st.caption("No events yet.")
    else:
        st.dataframe(events[::-1], use_container_width=True, hide_index=True)


# ----------------------------
# Module: Document Review
# ----------------------------

def document_review_page() -> None:
    st.header(TEXT[st.session_state["ui_lang"]]["tabs"][5])

    st.markdown("<div class='wow-panel'>", unsafe_allow_html=True)
    st.write("Upload/paste documents for analysis. PDFs are extracted via text layer (OCR not included here).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Submission Documents (510(k) doc, summary, notes)")
    up = st.file_uploader("Upload submission docs (txt/md/pdf)", type=["txt", "md", "pdf"], accept_multiple_files=True)
    paste = st.text_area("Or paste submission text", height=180, placeholder="Paste 510(k) summary, review notes, or submission excerpt...")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Add Submission Inputs", use_container_width=True):
            added = 0
            if up:
                for f in up:
                    doc = normalize_uploaded_file(f)
                    doc["doc_type"] = "510k_submission"
                    st.session_state["docs"]["submission_docs"].append(doc)
                    added += 1
            if paste.strip():
                doc = normalize_pasted_text(paste.strip(), doc_type="510k_submission", source="paste")
                st.session_state["docs"]["submission_docs"].append(doc)
                added += 1
            log_event("docs", {"action": "add_submission_docs", "count": added})
            st.success(f"Added {added} submission inputs.")
    with colB:
        if st.button("Clear Submission Docs", use_container_width=True):
            st.session_state["docs"]["submission_docs"] = []
            log_event("docs", {"action": "clear_submission_docs"})
            st.warning("Cleared submission documents.")

    if st.session_state["docs"]["submission_docs"]:
        st.write("Current submission docs:")
        st.dataframe(
            [{
                "doc_id": d["doc_id"],
                "source": d["source"],
                "filename": d.get("original_filename"),
                "method": d.get("extraction_method"),
                "lang": d.get("language_detected"),
                "warnings": "; ".join(d.get("extraction_warnings") or []),
                "chars": len(d.get("extracted_text") or ""),
            } for d in st.session_state["docs"]["submission_docs"]],
            use_container_width=True,
            hide_index=True
        )

        with st.expander("Preview (first 2,000 chars combined)", expanded=False):
            preview = "\n\n---\n\n".join([(d.get("extracted_text") or "")[:2000] for d in st.session_state["docs"]["submission_docs"][:3]])
            st.text(preview)

    st.divider()
    st.subheader("Guidance Document")
    use_default = st.checkbox("Use default guidance", value=st.session_state.get("default_guidance_enabled", True))
    st.session_state["default_guidance_enabled"] = use_default

    g_up = st.file_uploader("Upload guidance (txt/md)", type=["txt", "md"], accept_multiple_files=False)
    g_paste = st.text_area("Or paste guidance text", height=160)

    if st.button("Set Guidance", use_container_width=True):
        guidance_doc = None
        if g_up is not None:
            guidance_doc = normalize_uploaded_file(g_up)
            guidance_doc["doc_type"] = "guidance"
        elif g_paste.strip():
            guidance_doc = normalize_pasted_text(g_paste.strip(), doc_type="guidance", source="paste")
        elif use_default:
            guidance_doc = normalize_pasted_text(DEFAULT_GUIDANCE_ZH_TW, doc_type="guidance", source="default")
        else:
            guidance_doc = None

        st.session_state["docs"]["guidance_doc"] = guidance_doc
        log_event("docs", {"action": "set_guidance", "source": (guidance_doc or {}).get("source")})
        st.success("Guidance set.")

    gd = st.session_state["docs"]["guidance_doc"]
    if gd:
        st.write("Current guidance:")
        st.json({
            "doc_id": gd["doc_id"],
            "source": gd["source"],
            "filename": gd.get("original_filename"),
            "lang": gd.get("language_detected"),
            "chars": len(gd.get("extracted_text") or "")
        })
        with st.expander("Guidance preview (first 2,000 chars)", expanded=False):
            st.text((gd.get("extracted_text") or "")[:2000])
    else:
        st.info("No guidance selected.")


# ----------------------------
# Module: 510(k) Report Generator
# ----------------------------

def build_report_context() -> str:
    docs = st.session_state["docs"]["submission_docs"]
    gd = st.session_state["docs"]["guidance_doc"]

    submission_text = "\n\n".join([d.get("extracted_text") or "" for d in docs]).strip()
    guidance_text = (gd.get("extracted_text") if gd else "") or ""

    # Keep context explicit. Real-world: you may chunk/summarize to stay within context.
    ctx = []
    ctx.append("# INPUT: 510(k) SUBMISSION DOCUMENT(S)\n")
    ctx.append(submission_text if submission_text else "[No submission content provided]")
    ctx.append("\n\n# INPUT: 510(k) REVIEW GUIDANCE\n")
    ctx.append(guidance_text if guidance_text else "[No guidance provided]")
    return "\n".join(ctx)


def report_generator_page() -> None:
    st.header(TEXT[st.session_state["ui_lang"]]["tabs"][1])

    ssr = st.session_state["report"]
    ensure_agents_loaded()  # ensures agents exist for later chaining

    # Configuration
    st.subheader("Configuration")

    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        ssr["output_lang"] = st.selectbox(
            "Output language",
            options=list(SUPPORTED_OUTPUT_LANGS.keys()),
            format_func=lambda k: SUPPORTED_OUTPUT_LANGS[k],
            index=list(SUPPORTED_OUTPUT_LANGS.keys()).index(ssr["output_lang"]),
        )
    with c2:
        ssr["provider"] = st.selectbox("Provider", options=PROVIDERS, index=PROVIDERS.index(ssr["provider"]))
    with c3:
        models = MODEL_CATALOG.get(ssr["provider"], [])
        if ssr["model"] not in models and models:
            ssr["model"] = models[0]
        ssr["model"] = st.selectbox("Model", options=models, index=models.index(ssr["model"]) if ssr["model"] in models else 0)

    c4, c5 = st.columns([1, 1])
    with c4:
        ssr["max_tokens"] = st.number_input("Max tokens", min_value=512, max_value=32000, value=int(ssr["max_tokens"]), step=256)
    with c5:
        ssr["temperature"] = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(ssr["temperature"]), step=0.05)

    ssr["prompt"] = st.text_area("Prompt (editable)", value=ssr["prompt"], height=260)

    st.subheader("Generate")
    st.caption("Tip: Load documents and guidance in the Document Review tab first for best results.")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Generate 510(k) Review Report", use_container_width=True):
            skill = build_skill_prompt()
            context = build_report_context()

            # Add output language instruction
            lang_clause = "Write the final report in English." if ssr["output_lang"] == "en" else "請以繁體中文撰寫最終報告。"
            system_prompt = skill + "\n\n" + "You must follow the user's requested structure and constraints.\n" + lang_clause

            user_prompt = ssr["prompt"] + "\n\n" + context

            log_event("report", {"action": "generate_start", "provider": ssr["provider"], "model": ssr["model"]})
            with st.spinner("Generating report…"):
                try:
                    out, meta = llm_generate(
                        provider=ssr["provider"],
                        model=ssr["model"],
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=int(ssr["max_tokens"]),
                        temperature=float(ssr["temperature"]),
                    )
                    ssr["markdown"] = out
                    ssr["last_run_meta"] = meta
                    log_event("report", {"action": "generate_success", "meta": {k: meta.get(k) for k in ("provider", "model", "latency_ms", "est_tokens_in", "est_tokens_out")}})
                    st.success("Report generated.")
                except Exception as e:
                    log_event("report", {"action": "generate_error", "error": str(e)})
                    st.error(str(e))

    with colB:
        if st.button("Use Report as Prompt Runner Input", use_container_width=True):
            st.session_state["runner"]["input_text"] = ssr["markdown"] or ""
            st.session_state["runner"]["use_report_as_input"] = False
            log_event("runner", {"action": "set_input_from_report"})
            st.info("Prompt Runner input updated from report output.")

    st.divider()
    st.subheader("Report Editor")
    if ssr["last_run_meta"]:
        st.caption(f"Last run: {ssr['last_run_meta'].get('provider')} / {ssr['last_run_meta'].get('model')} · {ssr['last_run_meta'].get('latency_ms')} ms")

    # Editable output
    ssr["markdown"] = st.text_area("Report (Markdown, editable)", value=ssr["markdown"], height=340, placeholder="Generate a report to see content here...")

    preview = st.checkbox("Preview Markdown", value=True)
    if preview and ssr["markdown"].strip():
        st.markdown("<div class='wow-panel'>", unsafe_allow_html=True)
        render_markdown(ssr["markdown"], allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Downloads
    st.subheader("Download")
    md_bytes = (ssr["markdown"] or "").encode("utf-8")
    st.download_button("Download .md", data=md_bytes, file_name="510k_review_report.md", mime="text/markdown", use_container_width=True)
    st.download_button("Download .txt", data=md_bytes, file_name="510k_review_report.txt", mime="text/plain", use_container_width=True)

    st.divider()
    st.subheader("Keep Prompting on the Report")
    followup = st.text_area("Instruction", height=120, placeholder="e.g., Add a Software / Cybersecurity section aligned to IEC 62304 and FDA guidance; do not invent evidence.")
    c1, c2, c3 = st.columns([1.1, 1.1, 1.1])
    with c1:
        p2 = st.selectbox("Provider (chat)", options=PROVIDERS, index=PROVIDERS.index(ssr["provider"]), key="report_chat_provider")
    with c2:
        m2 = st.selectbox("Model (chat)", options=MODEL_CATALOG.get(p2, []), index=0, key="report_chat_model")
    with c3:
        mt2 = st.number_input("Max tokens (chat)", min_value=256, max_value=32000, value=4000, step=256, key="report_chat_maxtokens")

    if st.button("Apply Instruction to Report", use_container_width=True):
        if not ssr["markdown"].strip():
            st.error("No report content to refine.")
        elif not followup.strip():
            st.error("Instruction is empty.")
        else:
            skill = build_skill_prompt()
            system_prompt = skill + "\n\nYou are refining an existing 510(k) review report. Preserve structure unless asked. Never invent evidence."
            user_prompt = f"## CURRENT REPORT\n{ssr['markdown']}\n\n## INSTRUCTION\n{followup}\n\nReturn the FULL updated report in Markdown."
            with st.spinner("Refining report…"):
                try:
                    out, meta = llm_generate(p2, m2, system_prompt, user_prompt, max_tokens=int(mt2), temperature=0.2)
                    ssr["markdown"] = out
                    ssr["last_run_meta"] = meta
                    log_event("report", {"action": "refine_success", "meta": {k: meta.get(k) for k in ("provider", "model", "latency_ms")}})
                    st.success("Report updated.")
                except Exception as e:
                    log_event("report", {"action": "refine_error", "error": str(e)})
                    st.error(str(e))


# ----------------------------
# Module: AI Note Keeper
# ----------------------------

NOTE_MAGICS = {
    "Auto-Structure": "Transform the note into organized Markdown with clear headings, bullets, action items, and open questions. Highlight key terms using **bold**.",
    "Executive Summary": "Write an executive summary of the note, then list top 10 risks/concerns. Use Markdown. Highlight key terms using **bold**.",
    "Risk Flagging": "Identify compliance/regulatory risks. Provide a risk table with severity and mitigation suggestions. Use Markdown. Highlight key terms using **bold**.",
    "Predicate Matcher": "Suggest predicate device categories and comparison dimensions (non-authoritative). Provide a comparison checklist. Use Markdown. Highlight key terms using **bold**.",
    "Table Extractor": "Extract structured data and format as Markdown tables. If none, propose sensible tables. Highlight key terms using **bold**.",
    "Medical Jargon Polish": "Rewrite using professional regulatory language, preserving meaning. Use Markdown. Highlight key terms using **bold**.",
}


def note_keeper_page() -> None:
    st.header(TEXT[st.session_state["ui_lang"]]["tabs"][3])

    ssn = st.session_state["note_keeper"]

    st.subheader("Input")
    up = st.file_uploader("Upload note (txt/md)", type=["txt", "md"], accept_multiple_files=False)
    if up is not None:
        doc = normalize_uploaded_file(up)
        ssn["input_text"] = doc.get("extracted_text") or ""
        log_event("notes", {"action": "upload_note", "filename": doc.get("original_filename")})

    ssn["input_text"] = st.text_area("Paste note", value=ssn["input_text"], height=220)

    st.subheader("Model")
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        ssn["provider"] = st.selectbox("Provider", options=PROVIDERS, index=PROVIDERS.index(ssn["provider"]), key="note_provider")
    with c2:
        ms = MODEL_CATALOG.get(ssn["provider"], [])
        if ssn["model"] not in ms and ms:
            ssn["model"] = ms[0]
        ssn["model"] = st.selectbox("Model", options=ms, index=ms.index(ssn["model"]) if ssn["model"] in ms else 0, key="note_model")
    with c3:
        ssn["max_tokens"] = st.number_input("Max tokens", min_value=512, max_value=32000, value=int(ssn["max_tokens"]), step=256, key="note_maxtokens")

    st.subheader("AI Magics")
    cols = st.columns(3)
    magic_names = list(NOTE_MAGICS.keys())
    for i, mn in enumerate(magic_names):
        with cols[i % 3]:
            if st.button(mn, use_container_width=True):
                if not ssn["input_text"].strip():
                    st.error("Input note is empty.")
                else:
                    skill = build_skill_prompt()
                    system_prompt = skill + "\n\nYou are an AI Note Keeper. Produce high-quality Markdown. Never invent facts."
                    user_prompt = f"{NOTE_MAGICS[mn]}\n\n## NOTE INPUT\n{ssn['input_text']}\n\nReturn Markdown only."
                    with st.spinner(f"Running: {mn}…"):
                        try:
                            out, meta = llm_generate(ssn["provider"], ssn["model"], system_prompt, user_prompt, max_tokens=int(ssn["max_tokens"]), temperature=0.2)
                            ssn["output_text"] = out
                            ssn["last_magic"] = mn
                            log_event("notes", {"action": "magic_success", "magic": mn, "meta": {k: meta.get(k) for k in ("provider", "model", "latency_ms")}})
                            st.success("Done.")
                        except Exception as e:
                            log_event("notes", {"action": "magic_error", "magic": mn, "error": str(e)})
                            st.error(str(e))

    st.divider()
    st.subheader("AI Keywords (user-defined, with color)")
    ssn["keywords_list"] = st.text_area("Keywords/phrases (one per line)", value=ssn["keywords_list"], height=120, placeholder="e.g.\nIEC 60601-1\nISO 10993\nSubstantial Equivalence")
    ssn["keywords_color"] = st.color_picker("Keyword color", value=ssn["keywords_color"])
    if st.button("Apply Keyword Highlighting to Output", use_container_width=True):
        kws = [x.strip() for x in (ssn["keywords_list"] or "").splitlines() if x.strip()]
        ssn["output_text"] = apply_keyword_spans(ssn["output_text"] or "", kws, ssn["keywords_color"])
        log_event("notes", {"action": "apply_keywords", "count": len(kws), "color": ssn["keywords_color"]})
        st.success("Keywords highlighted in output (HTML spans).")

    st.divider()
    st.subheader("Output Editor")
    ssn["output_text"] = st.text_area("Output (editable)", value=ssn["output_text"], height=280)

    if st.checkbox("Preview Output Markdown", value=True, key="note_preview"):
        st.markdown("<div class='wow-panel'>", unsafe_allow_html=True)
        render_markdown(ssn["output_text"], allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Keep Prompting on the Note/Output")
    instruction = st.text_area("Instruction (Note Keeper)", height=100, placeholder="e.g., Convert this into a meeting minute template with decisions, owners, and dates.")
    if st.button("Apply Instruction (append + refine)", use_container_width=True):
        if not instruction.strip():
            st.error("Instruction is empty.")
        else:
            skill = build_skill_prompt()
            system_prompt = skill + "\n\nYou refine note outputs. Preserve evidence discipline. Output Markdown."
            user_prompt = f"## CURRENT OUTPUT\n{ssn['output_text']}\n\n## INSTRUCTION\n{instruction}\n\nReturn FULL updated Markdown."
            with st.spinner("Refining…"):
                try:
                    out, meta = llm_generate(ssn["provider"], ssn["model"], system_prompt, user_prompt, max_tokens=4000, temperature=0.2)
                    ssn["output_text"] = out
                    log_event("notes", {"action": "refine_success", "meta": {k: meta.get(k) for k in ("provider", "model", "latency_ms")}})
                    st.success("Updated.")
                except Exception as e:
                    log_event("notes", {"action": "refine_error", "error": str(e)})
                    st.error(str(e))

    st.subheader("Download")
    out_bytes = (ssn["output_text"] or "").encode("utf-8")
    st.download_button("Download note .md", data=out_bytes, file_name="ai_note_keeper_output.md", mime="text/markdown", use_container_width=True)
    st.download_button("Download note .txt", data=out_bytes, file_name="ai_note_keeper_output.txt", mime="text/plain", use_container_width=True)


# ----------------------------
# Module: Agents YAML Manager
# ----------------------------

def agents_yaml_page() -> None:
    st.header(TEXT[st.session_state["ui_lang"]]["tabs"][4])

    ensure_agents_loaded()
    ss = st.session_state["agents_yaml"]

    st.subheader("Upload / Paste agents.yaml")
    up = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"], accept_multiple_files=False)
    if up is not None:
        raw = up.read().decode("utf-8", errors="replace")
        ss["raw"] = raw
        log_event("agents", {"action": "upload_yaml", "filename": up.name})

    ss["raw"] = st.text_area("agents.yaml (raw)", value=ss["raw"], height=260)

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        if st.button("Standardize YAML", use_container_width=True):
            std, rep, agents = standardize_agents_yaml(ss["raw"])
            ss["standardized"] = std
            ss["transform_report"] = rep
            ss["agents"] = agents
            log_event("agents", {"action": "standardize", "agents": len(agents)})
            st.success("Standardized.")
    with colB:
        if st.button("Load Default YAML", use_container_width=True):
            ss["raw"] = DEFAULT_AGENTS_YAML
            std, rep, agents = standardize_agents_yaml(ss["raw"])
            ss["standardized"] = std
            ss["transform_report"] = rep
            ss["agents"] = agents
            log_event("agents", {"action": "load_default", "agents": len(agents)})
            st.info("Loaded default.")
    with colC:
        dl = (ss["standardized"] or "").encode("utf-8")
        st.download_button("Download standardized agents.yaml", data=dl, file_name="agents.yaml", mime="text/yaml", use_container_width=True)

    st.divider()
    st.subheader("Transformation Report")
    st.markdown("<div class='wow-panel'>", unsafe_allow_html=True)
    render_markdown(ss["transform_report"] or "_No report yet._", allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Agents (Standardized View)")
    if ss["agents"]:
        st.dataframe(
            [{
                "id": a.get("id"),
                "name": a.get("name"),
                "provider": a.get("provider"),
                "model": a.get("model"),
                "max_tokens": a.get("max_tokens"),
                "temperature": a.get("temperature"),
                "tags": ", ".join((a.get("ui") or {}).get("tags", [])),
            } for a in ss["agents"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No agents loaded.")


# ----------------------------
# Module: Prompt Runner (Sequential Agents with Editable Handoffs)
# ----------------------------

def _build_runner_configs_from_agents(n: int) -> List[Dict[str, Any]]:
    agents = st.session_state["agents_yaml"]["agents"]
    if not agents:
        ensure_agents_loaded()
        agents = st.session_state["agents_yaml"]["agents"]

    chosen = agents[:n] if len(agents) >= n else (agents + agents[: max(0, n - len(agents))])
    cfgs = []
    for i, a in enumerate(chosen):
        cfgs.append({
            "step": i + 1,
            "agent_id": a["id"],
            "agent_name": a["name"],
            "provider": a["provider"],
            "model": a["model"],
            "max_tokens": a.get("max_tokens", DEFAULT_MAX_TOKENS),
            "temperature": a.get("temperature", DEFAULT_TEMPERATURE),
            "system_prompt": a.get("system_prompt", ""),
            "user_prompt": "",  # user can override; default built at run-time from input
            "output_mode": "markdown",
        })
    return cfgs


def prompt_runner_page() -> None:
    st.header(TEXT[st.session_state["ui_lang"]]["tabs"][2])

    ensure_agents_loaded()
    r = st.session_state["runner"]
    report_md = st.session_state["report"]["markdown"]

    st.subheader("Input")
    r["use_report_as_input"] = st.checkbox("Use current Report as input", value=r["use_report_as_input"])
    if r["use_report_as_input"] and report_md.strip():
        r["input_text"] = report_md
        st.info("Using report output as runner input.")
    else:
        r["input_text"] = st.text_area("Runner input text", value=r["input_text"], height=220)

    st.subheader("Agent Chain Setup")
    r["n_agents"] = st.slider("Number of agents", min_value=1, max_value=8, value=int(r["n_agents"]), step=1)

    if not r["configs"] or len(r["configs"]) != int(r["n_agents"]):
        r["configs"] = _build_runner_configs_from_agents(int(r["n_agents"]))
        r["outputs"] = [""] * int(r["n_agents"])
        r["logs"] = []
        log_event("runner", {"action": "reset_chain", "n_agents": r["n_agents"]})

    st.caption("You can edit prompt/model/max tokens before execution. After each step, you can edit output before passing to the next agent.")

    # Config editors
    for i, cfg in enumerate(r["configs"]):
        with st.expander(f"Step {cfg['step']}: {cfg['agent_name']} ({cfg['agent_id']})", expanded=(i == 0)):
            c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
            with c1:
                cfg["provider"] = st.selectbox("Provider", options=PROVIDERS, index=PROVIDERS.index(cfg["provider"]), key=f"runner_provider_{i}")
            with c2:
                ms = MODEL_CATALOG.get(cfg["provider"], [])
                if cfg["model"] not in ms and ms:
                    cfg["model"] = ms[0]
                cfg["model"] = st.selectbox("Model", options=ms, index=ms.index(cfg["model"]) if cfg["model"] in ms else 0, key=f"runner_model_{i}")
            with c3:
                cfg["max_tokens"] = st.number_input("Max tokens", min_value=512, max_value=32000, value=int(cfg["max_tokens"]), step=256, key=f"runner_maxtokens_{i}")

            cfg["temperature"] = st.slider("Temperature", 0.0, 1.0, float(cfg["temperature"]), 0.05, key=f"runner_temp_{i}")

            st.caption("System prompt (from agents.yaml + SKILL.md policy at runtime):")
            st.text_area("System prompt (editable if you choose)", value=cfg["system_prompt"], height=120, key=f"runner_sysp_{i}")

            cfg["user_prompt"] = st.text_area(
                "User prompt override (optional)",
                value=cfg["user_prompt"],
                height=100,
                placeholder="Leave blank to use the previous step output as the main input with a default instruction.",
                key=f"runner_userp_{i}"
            )

    st.divider()
    st.subheader("Run")
    if st.button("Run Agents Sequentially", use_container_width=True):
        if not r["input_text"].strip():
            st.error("Runner input is empty.")
        else:
            skill = build_skill_prompt()
            current_input = r["input_text"]
            r["logs"] = []
            r["outputs"] = [""] * int(r["n_agents"])

            for i, cfg in enumerate(r["configs"]):
                step = cfg["step"]
                agent_name = cfg["agent_name"]

                # Construct prompts
                system_prompt = skill + "\n\n" + (cfg["system_prompt"] or "")
                if cfg["user_prompt"].strip():
                    user_prompt = cfg["user_prompt"].strip() + "\n\n# INPUT\n" + current_input
                else:
                    user_prompt = (
                        f"You are running agent '{agent_name}'.\n"
                        "Process the INPUT below and produce the agent's required output in Markdown.\n"
                        "Never invent evidence; label missing information as Not provided.\n\n"
                        f"# INPUT\n{current_input}"
                    )

                # Execute
                r["logs"].append({"ts": dt.datetime.now().isoformat(timespec="seconds"), "step": step, "status": "running", "agent": agent_name})
                log_event("runner", {"action": "step_start", "step": step, "agent": agent_name, "provider": cfg["provider"], "model": cfg["model"]})

                with st.spinner(f"Running step {step}: {agent_name}…"):
                    try:
                        out, meta = llm_generate(
                            provider=cfg["provider"],
                            model=cfg["model"],
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            max_tokens=int(cfg["max_tokens"]),
                            temperature=float(cfg["temperature"]),
                        )
                        r["outputs"][i] = out
                        r["logs"].append({"ts": dt.datetime.now().isoformat(timespec="seconds"), "step": step, "status": "success", "agent": agent_name, "meta": {k: meta.get(k) for k in ("latency_ms", "est_tokens_in", "est_tokens_out")}})
                        log_event("runner", {"action": "step_success", "step": step, "agent": agent_name, "meta": {k: meta.get(k) for k in ("provider", "model", "latency_ms")}})
                    except Exception as e:
                        r["logs"].append({"ts": dt.datetime.now().isoformat(timespec="seconds"), "step": step, "status": "error", "agent": agent_name, "error": str(e)})
                        log_event("runner", {"action": "step_error", "step": step, "agent": agent_name, "error": str(e)})
                        st.error(f"Step {step} failed: {e}")
                        break

                # Editable handoff: set current_input to output (user can edit later in UI, too)
                current_input = r["outputs"][i]

            st.success("Run completed (or stopped on error).")

    st.divider()
    st.subheader("Outputs (Editable Handoffs)")

    for i, cfg in enumerate(r["configs"]):
        out = r["outputs"][i] if i < len(r["outputs"]) else ""
        with st.expander(f"Output — Step {cfg['step']}: {cfg['agent_name']}", expanded=(i == 0)):
            edited = st.text_area("Output (editable; becomes next input if you rerun from here)", value=out, height=220, key=f"runner_out_{i}")
            r["outputs"][i] = edited

            if st.checkbox("Preview Markdown", value=True, key=f"runner_prev_{i}"):
                st.markdown("<div class='wow-panel'>", unsafe_allow_html=True)
                render_markdown(edited, allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Set as Runner Input", use_container_width=True, key=f"runner_set_input_{i}"):
                r["input_text"] = edited
                r["use_report_as_input"] = False
                log_event("runner", {"action": "set_input_from_step", "step": cfg["step"]})
                st.info("Runner input updated from this output.")

    st.subheader("Run Log")
    if r["logs"]:
        st.dataframe(r["logs"][::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("No logs yet.")

    st.subheader("Download Combined Outputs")
    combined = []
    for i, cfg in enumerate(r["configs"]):
        combined.append(f"# Step {cfg['step']}: {cfg['agent_name']} ({cfg['agent_id']})\n\n{r['outputs'][i]}\n")
        combined.append("\n---\n")
    combined_md = "\n".join(combined).encode("utf-8")
    st.download_button("Download run book (.md)", data=combined_md, file_name="prompt_runner_run_book.md", mime="text/markdown", use_container_width=True)


# ----------------------------
# App Bootstrap
# ----------------------------

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    _init_state()
    inject_css()
    sidebar_controls()

    tabs = TEXT[st.session_state["ui_lang"]]["tabs"]
    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        dashboard_page()
    with tab_objs[1]:
        report_generator_page()
    with tab_objs[2]:
        prompt_runner_page()
    with tab_objs[3]:
        note_keeper_page()
    with tab_objs[4]:
        agents_yaml_page()
    with tab_objs[5]:
        document_review_page()

    # Footer
    st.divider()
    st.caption("Session-only app: keys and documents are stored in session memory unless you export downloads.")


if __name__ == "__main__":
    main()
