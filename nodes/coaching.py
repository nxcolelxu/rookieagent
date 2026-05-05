import json
from langchain_core.messages import AIMessage
from llm_client import get_llm
from state import RookieState

# TODO: RAG 연동 예정 - Vector DB에서 유사 직무 합격 자소서 패턴 조회 후 피드백에 반영
# TODO: UC6 연동 예정 - 자소서 내용이 user_profile과 일치하는지 확인 (사실 검증)
# TODO: UC7 연동 예정 - 생성된 피드백에 할루시네이션/모순이 없는지 검증
# TODO: coaching_round >= N 이면 "수준 양호" 판단 후 종료 제안 (N은 추후 결정)

SYSTEM_PROMPT = """당신은 취업 자소서 코칭 전문가입니다.
구직자의 자소서 단락을 JD와 비교 분석하여 실질적인 피드백을 제공합니다.

[평가 항목]
1. jd_fit_score (0.0~1.0): 자소서 내용이 JD 요구사항을 얼마나 반영하는가
2. job_fit_score (0.0~1.0): 경험/역량이 해당 직무에 적합한가
3. tone_score (0.0~1.0): 두괄식 구성, 문장 명확성, 구체적 수치 사용 여부
4. ai_detection_score (0.0~1.0): AI가 작성한 것처럼 보이는 정도 (낮을수록 좋음)

[피드백 원칙]
- "더 구체적으로 써주세요" 같은 모호한 피드백 절대 금지
- 어떤 표현을 어떻게 바꿔야 하는지 구체적으로 명시
- JD에서 어떤 키워드가 빠졌는지 missing_jd_keywords에 명시
- improved_example은 구직자의 실제 경험과 JD 키워드를 반영하여 작성
  (일반적인 예시 문장이 아닌, 해당 구직자에게 맞춤화된 예시)

[출력 규칙]
- 반드시 JSON 객체만 출력하라. 설명 텍스트, 마크다운 코드블록 일체 금지.

[출력 형식]
{
  "paragraph_id": "paragraph_1",
  "jd_fit_score": 0.00,
  "job_fit_score": 0.00,
  "tone_score": 0.00,
  "ai_detection_score": 0.00,
  "feedback_text": "구체적인 피드백 내용",
  "missing_jd_keywords": ["키워드1", "키워드2"],
  "improved_example": "개선된 예시 문장 (구직자 경험 반영)"
}"""


def _generate_paragraph_feedback(
    paragraph_id: str,
    paragraph_text: str,
    selected_job: dict,
    user_profile: dict,
) -> dict:
    """
    자소서 단락 1개에 대해 JD 기반 피드백을 생성한다.
    LLM에 단락 내용, JD 정보, 구직자 프로파일을 함께 전달한다.
    """
    llm = get_llm(temperature=0.3)

    feedback_prompt = f"""{SYSTEM_PROMPT}

[구직자 프로파일]
{json.dumps(user_profile, ensure_ascii=False, indent=2)}

[지원 공고 JD]
회사: {selected_job.get('company', '')}
직무: {selected_job.get('title', '')}
JD 요약: {selected_job.get('jd_summary', '')}
요구 스킬: {', '.join(selected_job.get('required_skills', []))}

[평가할 자소서 단락 ID]
{paragraph_id}

[평가할 자소서 단락 내용]
{paragraph_text}

위 자소서 단락에 대해 JD 기반 피드백을 JSON 형식으로 반환하라."""

    try:
        response = llm.invoke(feedback_prompt)
        raw = response.content.strip()
        # 마크다운 코드블록이 포함된 경우 제거
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        feedback = json.loads(raw.strip())
        # paragraph_id 보정 (LLM이 다르게 반환할 경우 대비)
        feedback["paragraph_id"] = paragraph_id
        return feedback
    except (json.JSONDecodeError, Exception) as e:
        print(f"[coaching] {paragraph_id} 피드백 파싱 실패: {e}")
        return {
            "paragraph_id": paragraph_id,
            "jd_fit_score": 0.0,
            "job_fit_score": 0.0,
            "tone_score": 0.0,
            "ai_detection_score": 0.0,
            "feedback_text": "피드백 생성 중 오류가 발생했습니다.",
            "missing_jd_keywords": [],
            "improved_example": ""
        }


def coaching_node(state: RookieState) -> dict:
    """
    UC3: 서류 코칭 에이전트 노드.
    cover_letter의 각 단락에 대해 selected_job JD 기반 피드백을 생성하고
    feedback_history에 라운드별로 누적 저장한다.
    """
    cover_letter = state.get("cover_letter", {})
    selected_job = state.get("selected_job", {})
    user_profile = state.get("user_profile", {})
    feedback_history = list(state.get("feedback_history", []))
    coaching_round = state.get("coaching_round", 0)

    # 자소서가 없는 경우 조기 종료
    if not cover_letter:
        print("[coaching] 자소서 없음 - 코칭 건너뜀")
        return {
            "current_step": "interview",
            "coaching_done": True,
            "messages": [AIMessage(content="자소서가 없어 코칭을 건너뜁니다.")],
        }

    # 각 단락별로 순차적으로 피드백 생성
    paragraph_feedbacks = []
    for paragraph_id, paragraph_text in cover_letter.items():
        print(f"[coaching] {paragraph_id} 피드백 생성 중...")
        feedback = _generate_paragraph_feedback(
            paragraph_id, paragraph_text, selected_job, user_profile
        )
        paragraph_feedbacks.append(feedback)

    # 이번 라운드 피드백을 history에 추가
    new_round = coaching_round + 1
    feedback_history.append({
        "round": new_round,
        "feedbacks": paragraph_feedbacks,
    })
    print(f"[coaching] 라운드 {new_round} 피드백 {len(paragraph_feedbacks)}개 생성 완료")

    # 피드백 요약 메시지 구성
    summary_lines = [f"[라운드 {new_round}] 자소서 코칭 결과예요!\n"]
    for fb in paragraph_feedbacks:
        pid = fb.get("paragraph_id", "")
        jd_score = int(fb.get("jd_fit_score", 0) * 100)
        job_score = int(fb.get("job_fit_score", 0) * 100)
        tone_score = int(fb.get("tone_score", 0) * 100)
        ai_score = int(fb.get("ai_detection_score", 0) * 100)
        summary_lines.append(
            f"▶ {pid}\n"
            f"  JD 적합도: {jd_score}% | 직무 적합도: {job_score}% | "
            f"어조: {tone_score}% | AI 위험도: {ai_score}%\n"
            f"  💬 {fb.get('feedback_text', '')}"
        )
    summary_msg = "\n\n".join(summary_lines)

    return {
        "feedback_history": feedback_history,
        "coaching_round": new_round,
        "current_step": "interview",
        "messages": [AIMessage(content=summary_msg)],
    }
