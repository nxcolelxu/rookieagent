import json
from graph import build_graph


def print_state_diff(step_name: str, state: dict):
    """노드 실행 후 주요 State 변화를 출력한다."""
    print(f"\n{'='*50}")
    print(f"[{step_name}] 완료 후 State 요약")
    print(f"{'='*50}")
    print(f"current_step     : {state.get('current_step')}")
    print(f"profiling_stage  : {state.get('profiling_stage')}")
    print(f"profile_confirmed: {state.get('profile_confirmed')}")
    print(f"matched_jobs 수  : {len(state.get('matched_jobs', []))}")
    print(f"coaching_round   : {state.get('coaching_round')}")
    print(f"feedback_history : {len(state.get('feedback_history', []))} 라운드")
    print(f"interview_q 수   : {len(state.get('interview_questions', []))}")
    print(f"interview_qa 수  : {len(state.get('interview_qa', []))}")
    print(f"interview_done   : {state.get('interview_done')}")
    print()


def main():
    # 그래프 빌드
    app = build_graph()

    # 테스트용 초기 상태
    initial_state = {
        # UC1 초기값
        "user_profile": {},
        "profile_summary": "",
        "profile_confirmed": False,
        "profiling_stage": "greeting",

        # UC2 초기값
        "matched_jobs": [],
        "selected_job": {
            # 테스트용 더미 선택 공고 (실제로는 구직자가 matched_jobs에서 선택)
            "id": "job_001",
            "company": "카카오",
            "title": "백엔드 개발자 (신입)",
            "jd_summary": "Python/Java 기반 서버 개발, RESTful API 설계, DB 최적화",
            "required_skills": ["Python", "Java", "RESTful API", "MySQL"]
        },

        # UC3 초기값
        "cover_letter": {
            "paragraph_1": (
                "저는 문제 해결을 즐기는 개발자입니다. "
                "스타트업 인턴 시절 레거시 API의 응답속도를 30% 개선한 경험이 있습니다. "
                "이 과정에서 데이터베이스 쿼리 최적화와 캐싱 전략의 중요성을 배웠습니다."
            ),
            "paragraph_2": (
                "팀 프로젝트에서 백엔드를 담당하며 협업의 중요성을 배웠습니다. "
                "Spring Boot로 REST API를 설계하고 MySQL을 연동한 경험이 있습니다."
            )
        },
        "feedback_history": [],
        "coaching_round": 0,
        "coaching_done": False,

        # UC4 초기값
        "interview_questions": [],
        "interview_qa": [],
        "interview_question_count": 5,  # 테스트용 5개 (실제 기본값: 10)
        "interview_done": False,

        # 공통
        "current_step": "profiling",
        "messages": []
    }

    print("루키 AI 메인 플로우 테스트 시작")
    print(f"초기 current_step: {initial_state['current_step']}")

    # 그래프 실행
    final_state = app.invoke(initial_state)

    # 최종 결과 출력
    print("\n" + "="*50)
    print("최종 State 출력")
    print("="*50)
    print(json.dumps(
        {k: v for k, v in final_state.items() if k != "messages"},
        ensure_ascii=False,
        indent=2,
        default=str
    ))


if __name__ == "__main__":
    main()
