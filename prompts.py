# prompts.py
def build_prompt(profile, user_input, context_snippets=None):
    sys = (
      f"너는 상황특화 챗봇이다. 상황: {profile['description']}\n"
      f"말투: {profile['tone']}\n"
      f"목표: {', '.join(profile['goals'])}\n"
      f"해야할 일: {', '.join(profile['do'])}\n"
      f"금지: {', '.join(profile['dont'])}\n"
      f"응답은 한국어. 꺾쇠(<,>)나 placeholder를 쓰지 말고 실제 문장으로 채워라."
    )
    ctx = "\n\n[상황 문맥]\n" + "\n".join(context_snippets) if context_snippets else ""
    user = f"[사용자 입력]\n{user_input}"

    # few-shot 예시 (모델이 형식을 '채워서' 쓰도록 유도)
    example = (
      "=== 예시 ===\n"
      "메시지: 지금은 바쁘니 다음에 방문해 주세요. 집 안에 가족이 쉬고 있습니다.\n"
      "옵션:\n"
      "- 누구시죠? 용건을 문 아래로 남겨 주세요.\n"
      "- 가족이 자고 있어 응대가 어렵습니다. 연락처를 남겨 주세요.\n"
      "- 경비실에 확인하겠습니다.\n"
      "비고: 문은 열지 말고 대화는 짧게. 불안하면 112/경비실 연락.\n"
      "=== 예시 끝 ==="
    )

    inst = (
      "아래 형식을 정확히 따르되 모든 항목을 실제 내용으로 채워라.\n"
      "형식:\n"
      "메시지: (핵심 응답 한 단락)\n"
      "옵션:\n"
      "- (선택지1)\n"
      "- (선택지2)\n"
      "- (선택지3)\n"
      "비고: (위험/주의/다음 단계)"
    )
    return f"{sys}\n{ctx}\n\n{example}\n\n{user}\n\n{inst}"
