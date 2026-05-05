import json
from langchain_core.messages import HumanMessage, AIMessage
from llm_client import get_llm
from state import RookieState

# TODO: UC6 연동 예정 - 프로파일 입력 완료 후 모호한 정보 탐지 및 보완 질문 (최대 3개)
# TODO: UC7 연동 예정 - 프로파일 요약문에 할루시네이션/모순이 없는지 검증

SYSTEM_PROMPT = """당신은 취준생의 든든한 취업 도우미 "루키"입니다.
친근하고 따뜻한 말투로 대화하며, 취준생이 부담 없이 자신의 스펙을 이야기할 수 있도록 돕습니다.

[대화 원칙]
- 존댓말을 사용하되 딱딱하지 않게. ("~이에요", "~군요!", "대단한걸요?" 등 자연스러운 표현 사용)
- 취준생의 답변을 공감하며 긍정적으로 반응한 뒤, 자연스럽게 다음 질문으로 이어가기
- 한 번에 반드시 하나의 항목만 질문하기 (여러 항목 동시 질문 금지)
- 답변이 모호하면 구체적으로 되물어보기
  예) "인턴 경험이 있으시군요! 어느 회사에서 얼마나 일하셨어요?"
- 수집이 완료된 항목은 다시 묻지 않기

[수집해야 할 항목 목록]
이름 → 학교/전공/학점/졸업상태 → 자격증 → 어학 → 경험(인턴/프로젝트/활동) → 희망직무 → 희망기업유형
(위 순서를 따르되, 대화 흐름상 자연스러운 순서로 유연하게 조정 가능)

[금지 사항]
- 여러 항목을 한꺼번에 묻는 것
- "입력해주세요", "작성해주세요" 등 양식 느낌의 표현
- 학점이 낮거나 스펙이 부족한 것에 대해 부정적으로 반응하는 것
- 구직자를 평가하거나 비교하는 발언"""


def _extract_profile_from_conversation(messages: list, current_profile: dict) -> dict:
    """
    대화 이력과 기존 프로파일을 LLM에 전달하여 최신 정보를 추출·업데이트한다.
    마지막 사용자 메시지에서 새로운 정보를 파싱하여 current_profile에 병합한다.
    """
    llm = get_llm(temperature=0.1)

    # 대화 이력을 텍스트로 변환
    conversation_text = "\n".join(
        f"{'사용자' if isinstance(m, HumanMessage) else '루키'}: {m.content}"
        for m in messages[-10:]  # 최근 10개 메시지만 사용
    )

    extraction_prompt = f"""아래 대화에서 구직자의 스펙 정보를 추출하여 JSON으로 반환하라.
기존 프로파일에 없는 새 정보만 추가하고, 기존 정보는 그대로 유지하라.

[기존 프로파일]
{json.dumps(current_profile, ensure_ascii=False)}

[최근 대화]
{conversation_text}

[추출 규칙]
- 반드시 JSON 객체만 출력하라. 설명 텍스트, 마크다운 코드블록 일체 금지.
- 새로 언급된 정보만 포함하라. 언급되지 않은 필드는 기존 값을 그대로 유지.
- 학점은 "3.8/4.5" 형식으로 정규화.
- experiences는 기존 목록에 새 항목을 append하라.

[출력 형식]
{{
  "name": "이름 (없으면 기존값 또는 null)",
  "education": {{"university": "", "major": "", "gpa": "", "graduation_status": ""}},
  "certificates": [],
  "languages": [],
  "experiences": [],
  "desired_job": "",
  "desired_company_type": ""
}}"""

    try:
        response = llm.invoke(extraction_prompt)
        extracted = json.loads(response.content.strip())
        # 기존 프로파일과 병합: None/빈 값은 기존 값 유지
        merged = dict(current_profile)
        for key, value in extracted.items():
            if value is not None and value != "" and value != [] and value != {}:
                if key == "experiences" and isinstance(value, list):
                    # 경험은 기존 목록에 새 항목 추가 (중복 방지)
                    existing = merged.get("experiences", [])
                    for exp in value:
                        if exp not in existing:
                            existing.append(exp)
                    merged["experiences"] = existing
                elif key == "education" and isinstance(value, dict):
                    existing_edu = merged.get("education", {})
                    for k, v in value.items():
                        if v:
                            existing_edu[k] = v
                    merged["education"] = existing_edu
                else:
                    merged[key] = value
        return merged
    except (json.JSONDecodeError, Exception) as e:
        print(f"[profiling] 프로파일 추출 실패: {e}")
        return current_profile


def _is_profile_complete(profile: dict) -> bool:
    """
    필수 항목이 모두 채워졌는지 확인한다.
    이름, 학교 정보(대학/전공/졸업상태), 희망직무, 희망기업유형이 최소 기준.
    """
    if not profile.get("name"):
        return False
    edu = profile.get("education", {})
    if not edu.get("university") or not edu.get("major") or not edu.get("graduation_status"):
        return False
    if not profile.get("desired_job"):
        return False
    if not profile.get("desired_company_type"):
        return False
    return True


def _generate_next_question(profile: dict, messages: list) -> str:
    """
    현재까지 수집된 프로파일을 기반으로 가장 자연스러운 다음 질문 1개를 생성한다.
    """
    llm = get_llm(temperature=0.7)

    # 아직 수집되지 않은 항목 파악
    missing_items = []
    if not profile.get("name"):
        missing_items.append("이름")
    edu = profile.get("education", {})
    if not edu.get("university"):
        missing_items.append("학교명")
    if not edu.get("major"):
        missing_items.append("전공")
    if not edu.get("gpa"):
        missing_items.append("학점")
    if not edu.get("graduation_status"):
        missing_items.append("졸업상태(졸업/재학/졸업예정)")
    if not profile.get("certificates"):
        missing_items.append("자격증")
    if not profile.get("languages"):
        missing_items.append("어학 성적")
    if not profile.get("experiences"):
        missing_items.append("경험(인턴/프로젝트/활동)")
    if not profile.get("desired_job"):
        missing_items.append("희망 직무")
    if not profile.get("desired_company_type"):
        missing_items.append("희망 기업 유형")

    # 최근 대화 이력 구성
    conversation_text = "\n".join(
        f"{'사용자' if isinstance(m, HumanMessage) else '루키'}: {m.content}"
        for m in messages[-6:]
    )

    question_prompt = f"""{SYSTEM_PROMPT}

[현재까지 수집된 정보]
{json.dumps(profile, ensure_ascii=False)}

[아직 수집되지 않은 항목]
{', '.join(missing_items) if missing_items else '없음'}

[최근 대화]
{conversation_text}

위 대화 흐름에 이어서, 아직 수집되지 않은 항목 중 가장 자연스러운 항목 1개에 대한 질문을 생성하라.
- 이전 답변에 공감하는 짧은 반응(1문장)과 함께 질문 1개만 작성하라.
- 질문 외 다른 설명은 절대 포함하지 말 것."""

    try:
        response = llm.invoke(question_prompt)
        return response.content.strip()
    except Exception as e:
        print(f"[profiling] 다음 질문 생성 실패: {e}")
        return f"아직 {missing_items[0] if missing_items else '정보'}을 알려주시겠어요?"


def _extract_job_conditions(messages: list, current_conditions: list) -> list:
    """
    마지막 사용자 메시지에서 필수 조건을 추출하여 구조화된 목록으로 반환한다.
    "없어요" 등 조건 없음 응답이면 빈 리스트를 반환한다.
    """
    llm = get_llm(temperature=0.1)

    # 가장 최근 사용자 메시지만 사용
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    if not last_user_msg:
        return current_conditions

    extraction_prompt = f"""구직자가 언급한 취업 필수 조건을 추출하여 JSON 배열로 반환하라.

[구직자 발언]
{last_user_msg}

[추출 규칙]
- 반드시 JSON 배열만 출력하라. 설명 텍스트, 마크다운 코드블록 일체 금지.
- "없어요", "딱히 없어요", "상관없어요" 등 조건이 없다는 표현이면 빈 배열 [] 반환.
- category는 아래 중 하나로 분류: "근무지" | "근무형태" | "근무시간" | "연봉" | "복지" | "기타"
- condition은 공고 필터링에 사용할 수 있도록 명확하게 정규화하라.

[출력 형식]
[
  {{"category": "근무지", "condition": "서울 근무 필수", "raw": "서울에서 근무해야 해요"}}
]"""

    try:
        response = llm.invoke(extraction_prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        extracted = json.loads(raw.strip())
        # 기존 조건과 병합 (중복 제거)
        merged = list(current_conditions)
        for cond in extracted:
            if cond not in merged:
                merged.append(cond)
        return merged
    except (json.JSONDecodeError, Exception) as e:
        print(f"[profiling] 조건 추출 실패: {e}")
        return current_conditions


def _generate_profile_summary(profile: dict) -> str:
    """
    수집된 user_profile을 바탕으로 구직자에게 보여줄 한국어 요약문을 생성한다.
    """
    llm = get_llm(temperature=0.3)

    summary_prompt = f"""아래 구직자 프로파일을 바탕으로 친근하고 따뜻한 한국어 요약문을 작성하라.

[프로파일]
{json.dumps(profile, ensure_ascii=False, indent=2)}

[작성 원칙]
- 구직자에게 직접 말하는 2인칭 형식 ("OO님은 ~이에요")
- 핵심 스펙을 빠짐없이 언급
- 마지막에 "이 내용이 맞나요? 수정할 부분이 있으면 말씀해 주세요!" 문장 추가
- 400자 이내로 작성"""

    try:
        response = llm.invoke(summary_prompt)
        return response.content.strip()
    except Exception as e:
        print(f"[profiling] 요약문 생성 실패: {e}")
        return f"{profile.get('name', '고객')}님의 프로파일이 수집되었습니다. 이 내용이 맞나요?"


def profiling_node(state: RookieState) -> dict:
    """
    UC1: 프로파일링 에이전트 노드.
    대화 단계(profiling_stage)에 따라 분기하여 취준생 스펙을 수집하고 프로파일을 완성한다.
    """
    stage = state.get("profiling_stage", "greeting")
    messages = state.get("messages", [])
    user_profile = state.get("user_profile", {})
    updates = {}

    # ── greeting 단계: 첫 인사 및 이름 요청 ──────────────────────────────
    if stage == "greeting":
        greeting_msg = (
            "안녕하세요! 저는 취업 도우미 루키예요 😊\n"
            "취업 준비하시면서 힘드신 점 많으시죠? "
            "제가 딱 맞는 공고 찾기부터 자소서 코칭, 면접 준비까지 함께 도와드릴게요!\n\n"
            "먼저 편하게 얘기 나눠볼까요? 성함이 어떻게 되세요?"
        )
        updates["messages"] = [AIMessage(content=greeting_msg)]
        updates["profiling_stage"] = "collecting"
        updates["current_step"] = "profiling"
        print(f"[profiling] greeting → collecting 전환")
        return updates

    # ── collecting 단계: 사용자 답변에서 정보 추출 후 다음 질문 생성 ────────
    if stage == "collecting":
        # 마지막 메시지가 사용자 메시지인 경우에만 정보 추출
        if messages and isinstance(messages[-1], HumanMessage):
            # LLM으로 사용자 답변에서 프로파일 정보 추출
            user_profile = _extract_profile_from_conversation(messages, user_profile)
            updates["user_profile"] = user_profile

        # 모든 필수 항목이 수집되었는지 확인
        if _is_profile_complete(user_profile):
            # 요약 단계로 전환
            summary = _generate_profile_summary(user_profile)
            updates["profile_summary"] = summary
            updates["profiling_stage"] = "summarizing"
            updates["messages"] = [AIMessage(content=summary)]
            print(f"[profiling] collecting → summarizing 전환")
        else:
            # 다음 질문 생성
            next_question = _generate_next_question(user_profile, messages)
            updates["messages"] = [AIMessage(content=next_question)]
            print(f"[profiling] collecting 단계 - 다음 질문 생성")

        return updates

    # ── summarizing 단계: 구직자 확인 후 done으로 전환 ─────────────────────
    if stage == "summarizing":
        # 마지막 사용자 메시지가 확인/동의 표현인지 판단
        if messages and isinstance(messages[-1], HumanMessage):
            user_response = messages[-1].content.lower()
            # 확인 키워드 감지 (간단한 규칙 기반)
            confirm_keywords = ["맞아", "맞아요", "네", "예", "맞습니다", "좋아요", "좋아", "확인", "ok", "ㅇㅇ"]
            is_confirmed = any(kw in user_response for kw in confirm_keywords)

            if is_confirmed:
                # 프로파일 확정 → 필수 조건 수집 단계로 전환
                conditions_question = (
                    "완벽해요! 프로파일이 확정되었어요 🎉\n\n"
                    "마지막으로 한 가지만 더 여쭤볼게요!\n"
                    "공고를 찾을 때 꼭 지켜져야 하는 조건이 있나요? "
                    "예를 들면 '서울에서만 근무할 수 있어요', '재택근무가 필수예요', "
                    "'야근이 없어야 해요' 같은 것들이요.\n"
                    "없으시면 '없어요'라고 말씀해 주세요!"
                )
                updates["messages"] = [AIMessage(content=conditions_question)]
                updates["profiling_stage"] = "conditions"
                print(f"[profiling] summarizing → conditions 전환")
            else:
                # 수정 요청으로 간주하여 다시 수집 단계로
                retry_msg = (
                    "물론이죠! 어떤 부분을 수정할까요? "
                    "수정하실 내용을 말씀해 주시면 바로 반영해 드릴게요."
                )
                updates["messages"] = [AIMessage(content=retry_msg)]
                updates["profiling_stage"] = "collecting"
                print(f"[profiling] summarizing → collecting 전환 (수정 요청)")
        return updates

    # ── conditions 단계: 필수 조건 수집 ──────────────────────────────────
    if stage == "conditions":
        if messages and isinstance(messages[-1], HumanMessage):
            # 사용자 답변에서 필수 조건 추출
            job_conditions = _extract_job_conditions(
                messages, state.get("job_conditions", [])
            )

            if job_conditions:
                cond_list = "\n".join(f"  • [{c['category']}] {c['condition']}" for c in job_conditions)
                confirm_msg = (
                    f"알겠어요! 아래 조건을 기준으로 공고를 필터링할게요:\n{cond_list}\n\n"
                    "이제 딱 맞는 채용 공고를 찾아볼게요. 잠깐만 기다려 주세요!"
                )
            else:
                confirm_msg = (
                    "네, 별도 조건 없이 모든 공고를 대상으로 찾아볼게요!\n"
                    "잠깐만 기다려 주세요!"
                )

            updates["job_conditions"] = job_conditions
            updates["profiling_stage"] = "done"
            updates["profile_confirmed"] = True
            updates["current_step"] = "matching"
            updates["messages"] = [AIMessage(content=confirm_msg)]
            print(f"[profiling] conditions → done 전환, 조건 {len(job_conditions)}개 수집")
        else:
            # 아직 사용자 답변 없음 → 조건 질문 출력
            conditions_question = (
                "마지막으로 한 가지만 더 여쭤볼게요!\n"
                "공고를 찾을 때 꼭 지켜져야 하는 조건이 있나요? "
                "예를 들면 '서울에서만 근무할 수 있어요', '재택근무가 필수예요', "
                "'야근이 없어야 해요' 같은 것들이요.\n"
                "없으시면 '없어요'라고 말씀해 주세요!"
            )
            updates["messages"] = [AIMessage(content=conditions_question)]
            print(f"[profiling] conditions 단계 - 조건 질문 출력")
        return updates

    # ── done 단계: 이미 완료된 상태 ──────────────────────────────────────
    if stage == "done":
        updates["profile_confirmed"] = True
        updates["current_step"] = "matching"
        return updates

    return updates
