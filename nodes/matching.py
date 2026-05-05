import json
from langchain_core.messages import AIMessage
from llm_client import get_llm
from state import RookieState

# TODO: 실시간 공고 크롤링 연동 예정 (사람인/잡코리아 API) - DUMMY_JOB_POSTINGS 대체
# TODO: UC7 연동 예정 - 매칭 결과 및 점수 근거에 할루시네이션이 없는지 검증
# TODO: 매칭 후보가 N개 미만일 경우 검색 범위(직무 카테고리/지역) 자동 확장 로직

# 실제 크롤링 연동 전까지 사용하는 더미 공고 데이터
DUMMY_JOB_POSTINGS = [
    {
        "id": "job_001",
        "company": "카카오",
        "title": "백엔드 개발자 (신입)",
        "jd": (
            "Python 또는 Java 기반 서버 개발, RESTful API 설계 및 구현, "
            "관계형 DB 설계 및 최적화. "
            "우대: AWS/GCP 등 클라우드 플랫폼 경험, 대용량 트래픽 처리 경험"
        ),
        "required_skills": ["Python", "Java", "RESTful API", "MySQL", "클라우드"],
        "location": 0 # 0: seoul, 
    },
    {
        "id": "job_002",
        "company": "네이버",
        "title": "서버 개발자 (신입)",
        "jd": (
            "Spring Boot 기반 서버 개발, MySQL/Redis 사용 경험, "
            "대규모 데이터 처리 및 성능 최적화 경험. "
            "우대: 오픈소스 기여 경험, 알고리즘 문제해결 역량"
        ),
        "required_skills": ["Spring Boot", "Java", "MySQL", "Redis", "알고리즘"]
    },
    {
        "id": "job_003",
        "company": "라인플러스",
        "title": "백엔드 엔지니어 (신입/주니어)",
        "jd": (
            "Kotlin/Java 기반 MSA 설계 및 개발, Kafka/Redis 활용 경험, "
            "Docker/Kubernetes 기반 컨테이너 환경 이해. "
            "우대: 글로벌 서비스 운영 경험, 영어 커뮤니케이션 가능"
        ),
        "required_skills": ["Kotlin", "Java", "MSA", "Kafka", "Docker", "Kubernetes"]
    }
]

SYSTEM_PROMPT = """당신은 구직자의 스펙과 채용공고(JD)를 비교 분석하는 공고 매칭 전문가입니다.

[매칭 점수 산출 기준]
- 스펙 적합도 (40%): 학력/학점/자격증/어학이 JD 요구사항과 일치하는 정도
- JD 키워드 적합도 (35%): 희망직무와 JD 직무 설명의 키워드 겹침 정도
- 경험 유사도 (25%): 보유 경험(인턴/프로젝트 등)이 JD가 원하는 경험과 유사한 정도
위 기준의 가중합으로 최종 match_score를 0.0~1.0 사이로 산출하라.

[출력 규칙]
- 반드시 JSON 배열만 출력하라. 설명 텍스트, 마크다운 코드블록, 기타 문자열 일체 금지.
- missing_skills는 required_skills 중 구직자 프로파일에 없는 것만 포함하라.
- match_reason은 "왜 이 점수인지" 구체적으로 2~3문장으로 설명하라.

[출력 형식]
[
  {
    "id": "job_001",
    "company": "회사명",
    "title": "직무명",
    "jd_summary": "JD 핵심 내용 2~3줄 요약",
    "match_score": 0.00,
    "match_reason": "매칭 근거를 구체적으로 설명",
    "required_skills": ["스킬1", "스킬2"],
    "missing_skills": ["없는스킬1"]
  }
]"""


def matching_node(state: RookieState) -> dict:
    """
    UC2: 공고 매칭 에이전트 노드.
    user_profile을 분석하여 DUMMY_JOB_POSTINGS 각 공고에 매칭 점수를 산출하고
    match_score 내림차순으로 정렬하여 matched_jobs에 저장한다.
    """
    # TODO: location은 사용자가 입력한 것만 필터링하게 구현, location 뿐만 아니라 학력같은 정보들도 더 추가해야 함
    llm = get_llm(temperature=0.1)
    user_profile = state.get("user_profile", {})

    # 공고 목록을 텍스트로 변환하여 LLM에 전달
    jobs_text = json.dumps(DUMMY_JOB_POSTINGS, ensure_ascii=False, indent=2)

    matching_prompt = f"""{SYSTEM_PROMPT}

[구직자 프로파일]
{json.dumps(user_profile, ensure_ascii=False, indent=2)}

[분석할 채용공고 목록]
{jobs_text}

위 구직자 프로파일과 각 채용공고를 비교하여 매칭 결과를 JSON 배열로 반환하라."""

    matched_jobs = []
    try:
        response = llm.invoke(matching_prompt)
        raw = response.content.strip()
        # 마크다운 코드블록이 포함된 경우 제거
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        matched_jobs = json.loads(raw.strip())
        print(f"[matching] {len(matched_jobs)}개 공고 매칭 완료")
    except (json.JSONDecodeError, Exception) as e:
        print(f"[matching] 매칭 결과 파싱 실패: {e}")
        matched_jobs = []

    # match_score 내림차순 정렬
    matched_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)

    # 매칭 결과를 메시지로 구성
    if matched_jobs:
        result_lines = ["공고 매칭이 완료됐어요! 아래 공고들을 확인해 보세요:\n"]
        for i, job in enumerate(matched_jobs, 1):
            score_pct = int(job.get("match_score", 0) * 100)
            result_lines.append(
                f"{i}. [{job['company']}] {job['title']} — 매칭률 {score_pct}%\n"
                f"   {job.get('jd_summary', '')}"
            )
        result_msg = "\n".join(result_lines)
    else:
        result_msg = "죄송해요, 현재 조건에 맞는 공고를 찾지 못했어요. 조건을 조정해볼까요?"

    return {
        "matched_jobs": matched_jobs,
        "current_step": "coaching",
        "messages": [AIMessage(content=result_msg)],
    }
