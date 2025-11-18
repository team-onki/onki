import uvicorn, yaml, re
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch
from rag import SimpleRAG
from prompts import build_prompt
from huggingface_hub import login

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

import re

FALLBACK_OPTIONS = {
    "home_safety": [
        "누구시죠? 용건을 문 아래로 남겨 주세요.",
        "가족이 쉬고 있어 응대가 어렵습니다. 연락처를 남겨 주세요.",
        "경비실(관리사무소)에 확인하겠습니다."
    ],
    "cs_refund": [
        "주문번호와 구매일을 알려 주세요.",
        "제품 사용/포장 상태를 알려 주세요.",
        "수거지와 연락 가능 시간을 남겨 주세요."
    ],
}

def _strip_placeholders(text: str) -> str:
    # <...> 같은 placeholder를 제거
    return re.sub(r"<[^>]+>", "", text).strip()

def format_reply(raw: str, situation: str) -> str:
    # 프롬프트 부분 잘라냈다면 raw는 모델 응답만
    txt = _strip_placeholders(raw)
    # 섹션 추출
    m_msg = re.search(r"메시지:\s*(.+)", txt)
    m_opts = re.search(r"옵션:\s*(.*?)(?:\n\s*비고:|\Z)", txt, flags=re.S)
    m_note = re.search(r"비고:\s*(.+)", txt)

    msg = (m_msg.group(1).strip() if m_msg else "").strip()
    note = (m_note.group(1).strip() if m_note else "").strip()

    # 옵션 파싱
    opts = []
    if m_opts:
        for line in m_opts.group(1).splitlines():
            line = _strip_placeholders(line.strip())
            if line.startswith("-"):
                cand = line.lstrip("-").strip()
                if cand:
                    opts.append(cand)

    # 보정: 비어있으면 기본값 채움
    if not msg:
        msg = "지금은 응대가 어렵습니다. 다음에 방문해 주세요."
    while len(opts) < 3:
        # 상황별 기본 옵션 보충
        base = FALLBACK_OPTIONS.get(situation, [])
        for o in base:
            if o not in opts and len(opts) < 3:
                opts.append(o)
            if len(opts) >= 3:
                break
        if not base:  # 상황 기본값이 없으면 안전한 일반 옵션 사용
            opts.extend([
                "용건을 남겨 주세요.",
                "지금은 바쁩니다. 다음에 방문해 주세요.",
                "관리사무소에 확인하겠습니다."
            ])
            opts = opts[:3]

    if not note:
        note = "문은 열지 말고 대화는 짧게. 불안하면 112 또는 경비실 연락."

    # 최종 포맷
    return (
        f"메시지: {msg}\n"
        "옵션:\n"
        f"- {opts[0]}\n- {opts[1]}\n- {opts[2]}\n"
        f"비고: {note}"
    )


login('huggingface_token')

tokenizer = AutoTokenizer.from_pretrained('weights/gemma3')
model = AutoModelForCausalLM.from_pretrained(
    'weights/gemma3',
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()

with open("situations.yaml", "r", encoding="utf-8") as f:
    PROFILES = yaml.safe_load(f)["profiles"]

RAGS = {
    "home_safety": SimpleRAG("data/home_safety"),
    "cs_refund": SimpleRAG("data/cs_refund")
}

FORBIDDEN = re.compile(r"(주민등록번호|신용카드번호|정확한 집 주소|비밀번호)", re.I)

app = FastAPI(title="Gemma2B Situation Chatbot")

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3000/*",
    "http://localhost:3001",
    "http://localhost:3001/*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    situation: str     # 예: "home_safety" or "cs_refund"
    message: str

def guardrails(text: str) -> str:
    if FORBIDDEN.search(text):
        return "민감 정보는 안내할 수 없습니다. 다른 방식으로 설명해 주세요."
    return text

@app.post("/chat")
def chat(inp: ChatIn):
    if inp.situation not in PROFILES:
        return {"error": f"unknown situation: {inp.situation}"}
    profile = PROFILES[inp.situation]

    ctx_snippets = []
    try:
        if inp.situation in RAGS:
            ctx_snippets = RAGS[inp.situation].search(inp.message, k=3) or []
    except Exception as e:
        print(f"[RAG] search error: {e}")
        ctx_snippets = []
    
    prompt = build_prompt(profile, guardrails(inp.message), ctx_snippets)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=280,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    model_text = tokenizer.decode(out[0], skip_special_tokens=True)
    raw_reply = model_text[len(prompt):].strip()
    reply = format_reply(raw_reply, inp.situation)
    return {"reply": reply}
    
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)