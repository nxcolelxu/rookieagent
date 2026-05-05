import json
from langchain_core.messages import HumanMessage, AIMessage
from llm_client import get_llm
from state import RookieState

# TODO: UC7 연동 예정 - 생성된 면접 질문에 할루시네이션이 없는지 검증
# TODO: UC6 연동 예정 - 구직자 답변이 user_profile과 일치하는지 검증 (사실 모순 탐지)
# TODO: 중도 종료 감지 시 현재까지 결과로 부분 리포트 생성 (UC5 연계)
# TODO: 자소서 모순 탐지 시 사용자에게 명시적 노출 및 일관성 가이드 제공

QUESTION_SYSTEM_PROMPT = """당신은 실제 기업 면접관 역할을 하는 면접 시뮬레이터입니다.
구직자의 자소서와 JD를 분석하여 실전과 유사한 면접 질문을 생성합니다.

[질문 구성 비율] (총 질문 수 기준)
- 경험형 40%: 자소서의 구체적 경험을 근거로 묻는 질문
- 기술형 30%: JD 요구 기술 스택 및 직무 역량 관련 질문
- 상황형 20%: "만약 ~한다면 어떻게 하겠습니까?" 형태
- 압박형 10%: 지원자의 약점이나 불명확한 부분을 파고드는 질문

[출력 규칙]
- 반드시 JSON 배열만 출력하라. 설명 텍스트, 마크다운 코드블록 일체 금지.

[출력 형식]
[
  {
    "id": 1,
    "question": "질문 내용",
    "type": "경험형",
    "source": "자소서 기반"
  }
]"""

EVALUATION_SYSTEM_PROMPT = """당신은 면접관으로서 구직자의 답변을 평가하고 필요 시 꼬리질문을 생성합니다.

[평가 기준]
- answer_score (0.0~1.0): STAR 구조(Situation/Task/Action/Result) 충족 여부, 구체성, JD 연관성을 종합하여 평가
- has_follow_up: 답변에서 불명확하거나 더 깊이 파고들 포인트가 있으면 true
- follow_up_question: has_follow_up이 true일 때만 생성. 답변의 특정 부분을 지목하여 구체화 요청

[출력 규칙]
- 반드시 JSON 객체만 출력하라. 설명 텍스트, 마크다운 코드블록 일체 금지.

[출력 형식]
{
  "answer_score": 0.00,
  "feedback": "구체적인 답변 평가 내용",
  "has_follow_up": true,
  "follow_up_question": "꼬리질문 내용 (has_follow_up이 false이면 null)"
}"""


def _generate_questions(
    user_profile: dict,
    cover_letter: dict,
    selected_job: dict,
    question_count: int,
) -> list:
    """
    사용자 프로파일, 자소서, JD를 기반으로 면접 질문 question_count개를 생성한다.
    """
    llm = get_llm(temperature=0.7)

    question_prompt = f"""{QUESTION_SYSTEM_PROMPT}

[구직자 프로파일]
{json.dumps(user_profile, ensure_ascii=False, indent=2)}

[지원 공고 JD]
회사: {selected_job.get('company', '')}
직무: {selected_job.get('title', '')}
JD 요약: {selected_job.get('jd_summary', '')}
요구 스킬: {', '.join(selected_job.get('required_skills', []))}

[자소서]
{json.dumps(cover_letter, ensure_ascii=False, indent=2)}

위 정보를 분석하여 면접 질문 {question_count}개를 JSON 배열로 반환하라.
질문 구성 비율: 경험형 {round(question_count * 0.4)}개, 기술형 {round(question_count * 0.3)}개, 상황형 {round(question_count * 0.2)}개, 압박형 {round(question_count * 0.1)}개"""

    try:
        response = llm.invoke(question_prompt)
        raw = response.content.strip()
        # 마크다운 코드블록 제거
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        questions = json.loads(raw.strip())
        print(f"[interview] 면접 질문 {len(questions)}개 생성 완료")
        return questions
    except (json.JSONDecodeError, Exception) as e:
        print(f"[interview] 질문 생성 파싱 실패: {e}")
        return []


def _evaluate_answer(
    question: dict,
    answer: str,
    selected_job: dict,
) -> dict:
    """
    구직자의 답변을 평가하고 꼬리질문을 생성한다.
    STAR 구조, JD 연관성, 구체성을 기준으로 점수를 산출한다.
    """
    llm = get_llm(temperature=0.3)

    eval_prompt = f"""{EVALUATION_SYSTEM_PROMPT}

[면접 질문]
{question.get('question', '')} (유형: {question.get('type', '')})

[지원 공고]
회사: {selected_job.get('company', '')}, 직무: {selected_job.get('title', '')}
JD 요약: {selected_job.get('jd_summary', '')}

[구직자 답변]
{answer}

위 답변을 평가하고 꼬리질문 필요 여부를 JSON 형식으로 반환하라."""

    try:
        response = llm.invoke(eval_prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        evaluation = json.loads(raw.strip())
        return evaluation
    except (json.JSONDecodeError, Exception) as e:
        print(f"[interview] 답변 평가 파싱 실패: {e}")
        return {
            "answer_score": 0.0,
            "feedback": "평가 중 오류가 발생했습니다.",
            "has_follow_up": False,
            "follow_up_question": None,
        }


def interview_node(state: RookieState) -> dict:
    """
    UC4: 면접 시뮬레이션 에이전트 노드.
    - 질문 생성 모드: interview_questions가 비어있을 때 질문 목록 생성 후 첫 질문 출력
    - 답변 평가 모드: 사용자 답변이 들어왔을 때 평가 및 꼬리질문/다음 질문 진행
    """
    user_profile = state.get("user_profile", {})
    cover_letter = state.get("cover_letter", {})
    selected_job = state.get("selected_job", {})
    interview_questions = list(state.get("interview_questions", []))
    interview_qa = list(state.get("interview_qa", []))
    question_count = state.get("interview_question_count", 10)
    messages = state.get("messages", [])
    updates = {}

    # ── 질문 생성 모드: 아직 질문이 없을 때 ──────────────────────────────
    if not interview_questions:
        questions = _generate_questions(user_profile, cover_letter, selected_job, question_count)
        if not questions:
            return {
                "interview_done": True,
                "current_step": "done",
                "messages": [AIMessage(content="면접 질문 생성에 실패했습니다.")],
            }

        interview_questions = questions
        updates["interview_questions"] = interview_questions

        # 첫 번째 질문 출력
        first_q = interview_questions[0]
        intro_msg = (
            f"면접 시뮬레이션을 시작할게요! 총 {len(interview_questions)}개의 질문이 준비됐어요.\n\n"
            f"[질문 1/{len(interview_questions)}] ({first_q.get('type', '')} | {first_q.get('source', '')})\n"
            f"{first_q.get('question', '')}"
        )
        updates["messages"] = [AIMessage(content=intro_msg)]
        print(f"[interview] 질문 생성 모드 완료 - 첫 질문 출력")
        return updates

    # ── 답변 평가 모드: 사용자 답변이 들어왔을 때 ──────────────────────────
    # 현재 진행 중인 질문 인덱스 = 지금까지 완료된 QA 수
    current_q_idx = len(interview_qa)

    # 모든 질문이 완료된 경우
    if current_q_idx >= len(interview_questions):
        completion_msg = (
            "모든 면접 질문이 완료됐어요! 정말 수고하셨어요 🎉\n"
            "종합 피드백 리포트를 준비하고 있어요. 잠깐만 기다려 주세요."
        )
        return {
            "interview_done": True,
            "current_step": "done",
            "messages": [AIMessage(content=completion_msg)],
        }

    current_question = interview_questions[current_q_idx]

    # 마지막 메시지가 사용자 답변인 경우에만 평가 진행
    if not messages or not isinstance(messages[-1], HumanMessage):
        # 사용자 답변이 없으면 현재 질문을 다시 출력
        q_msg = (
            f"[질문 {current_q_idx + 1}/{len(interview_questions)}] "
            f"({current_question.get('type', '')} | {current_question.get('source', '')})\n"
            f"{current_question.get('question', '')}"
        )
        return {"messages": [AIMessage(content=q_msg)]}

    # 사용자 답변 평가
    user_answer = messages[-1].content
    evaluation = _evaluate_answer(current_question, user_answer, selected_job)
    print(f"[interview] 질문 {current_q_idx + 1} 답변 평가 완료 (점수: {evaluation.get('answer_score', 0):.2f})")

    # QA 이력에 추가
    qa_entry = {
        "question_id": current_question.get("id", current_q_idx + 1),
        "question": current_question.get("question", ""),
        "answer": user_answer,
        "follow_up_questions": [],
        "answer_score": evaluation.get("answer_score", 0.0),
        "feedback": evaluation.get("feedback", ""),
    }

    # 꼬리질문이 있는 경우
    if evaluation.get("has_follow_up") and evaluation.get("follow_up_question"):
        follow_up = evaluation["follow_up_question"]
        qa_entry["follow_up_questions"] = [follow_up]
        interview_qa.append(qa_entry)

        score_pct = int(evaluation.get("answer_score", 0) * 100)
        follow_up_msg = (
            f"답변 점수: {score_pct}점\n"
            f"💬 {evaluation.get('feedback', '')}\n\n"
            f"[꼬리질문]\n{follow_up}"
        )
        return {
            "interview_qa": interview_qa,
            "messages": [AIMessage(content=follow_up_msg)],
        }

    # 꼬리질문 없음 → 다음 질문으로 이동
    interview_qa.append(qa_entry)
    next_q_idx = len(interview_qa)

    score_pct = int(evaluation.get("answer_score", 0) * 100)
    feedback_msg = f"답변 점수: {score_pct}점\n💬 {evaluation.get('feedback', '')}"

    # 모든 질문 완료 확인
    if next_q_idx >= len(interview_questions):
        completion_msg = (
            f"{feedback_msg}\n\n"
            "모든 면접 질문이 완료됐어요! 정말 수고하셨어요 🎉\n"
            "종합 피드백 리포트를 준비하고 있어요."
        )
        return {
            "interview_qa": interview_qa,
            "interview_done": True,
            "current_step": "done",
            "messages": [AIMessage(content=completion_msg)],
        }

    # 다음 질문 출력
    next_question = interview_questions[next_q_idx]
    next_q_msg = (
        f"{feedback_msg}\n\n"
        f"[질문 {next_q_idx + 1}/{len(interview_questions)}] "
        f"({next_question.get('type', '')} | {next_question.get('source', '')})\n"
        f"{next_question.get('question', '')}"
    )
    return {
        "interview_qa": interview_qa,
        "messages": [AIMessage(content=next_q_msg)],
    }
