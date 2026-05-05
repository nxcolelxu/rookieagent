from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class RookieState(TypedDict):
    """
    루키 AI 전체 에이전트 파이프라인이 공유하는 상태 객체.
    UC1 → UC2 → UC3 → UC4 순서로 필드가 채워진다.
    """

    # ────────────────────────────────────────────
    # UC1: 프로파일링
    # ────────────────────────────────────────────

    user_profile: dict
    # 수집된 취준생 스펙. 최종 구조:
    # {
    #   "name": str,
    #   "education": {
    #       "university": str,
    #       "major": str,
    #       "gpa": str,
    #       "graduation_status": str          # "졸업" | "재학" | "졸업예정"
    #   },
    #   "certificates": list[str],
    #   "languages": list[dict],              # [{"name": "TOEIC", "score": "850"}]
    #   "experiences": list[dict],
    #       # {
    #       #   "type": str,                  # "인턴" | "프로젝트" | "동아리" | "대외활동"
    #       #   "organization": str,
    #       #   "duration": str,
    #       #   "description": str
    #       # }
    #   "desired_job": str,
    #   "desired_company_type": str
    # }

    profile_summary: str
    # 프로파일링 에이전트가 user_profile을 바탕으로 생성한 한국어 요약문.

    profile_confirmed: bool
    # True: 구직자가 프로파일 내용을 확인하고 확정함

    profiling_stage: str
    # "greeting" | "collecting" | "summarizing" | "conditions" | "done"

    job_conditions: list
    # 구직자가 중요시하는 필수 조건 목록. UC2 매칭 시 조건 위반 공고를 가중합 계산에서 제외한다.
    # 각 항목:
    # {
    #   "category": str,   # "근무지" | "근무형태" | "근무시간" | "연봉" | "복지" | "기타"
    #   "condition": str,  # 정규화된 조건 (예: "서울 근무 필수")
    #   "raw": str         # 사용자 원문 표현
    # }


    # ────────────────────────────────────────────
    # UC2: 공고 매칭
    # ────────────────────────────────────────────

    matched_jobs: list
    # 매칭된 공고 목록. 각 항목:
    # {
    #   "id": str, "company": str, "title": str,
    #   "jd_summary": str, "match_score": float,
    #   "match_reason": str, "required_skills": list[str],
    #   "missing_skills": list[str]
    # }

    selected_job: dict
    # 구직자가 최종 선택한 타겟 공고 1개.


    # ────────────────────────────────────────────
    # UC3: 자소서 코칭
    # ────────────────────────────────────────────

    cover_letter: dict
    # 자소서 단락별 내용. 키: 단락 ID, 값: 텍스트.

    feedback_history: list
    # 코칭 라운드별 피드백 이력.
    # [{"round": int, "feedbacks": list[dict]}]

    coaching_round: int
    # 현재까지 진행된 피드백 라운드 수.

    coaching_done: bool
    # True: 구직자가 최종 확정 선택


    # ────────────────────────────────────────────
    # UC4: 면접 시뮬레이션
    # ────────────────────────────────────────────

    interview_questions: list
    # 생성된 면접 질문 목록.
    # [{"id": int, "question": str, "type": str, "source": str}]

    interview_qa: list
    # 질문별 답변 및 평가 이력.
    # [{"question_id": int, "question": str, "answer": str,
    #   "follow_up_questions": list[str], "answer_score": float, "feedback": str}]

    interview_question_count: int
    # 면접에서 출제할 총 질문 개수. 기본값 10, 테스트 시 5.

    interview_done: bool
    # True: 모든 질문 완료


    # ────────────────────────────────────────────
    # 공통
    # ────────────────────────────────────────────

    current_step: str
    # "profiling" | "matching" | "coaching" | "interview" | "done"

    messages: Annotated[list, add_messages]
    # 전체 대화 이력. LangGraph add_messages reducer로 자동 누적.
