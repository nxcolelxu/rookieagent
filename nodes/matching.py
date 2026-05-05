import json
from langchain_core.messages import AIMessage
from llm_client import get_llm
from state import RookieState

# TODO: 실시간 공고 크롤링 연동 예정 (사람인/잡코리아 API) - DUMMY_JOB_POSTINGS 대체
# TODO: UC7 연동 예정 - 매칭 결과 및 점수 근거에 할루시네이션이 없는지 검증
# TODO: 매칭 후보가 N개 미만일 경우 검색 범위(직무 카테고리/지역) 자동 확장 로직

# 실제 크롤링 연동 전까지 사용하는 더미 공고 데이터
# location, work_type 필드는 job_conditions 필터링에 사용된다
DUMMY_JOB_POSTINGS = [
    {
        "id": "job_001",
        "company": "카카오",
        "title": "백엔드 개발자 (신입)",
        "location": "경기 성남시 판교 (본사)",
        "work_type": "하이브리드 (주 3일 출근)",
        "jd": (
            "Python 또는 Java 기반 서버 개발, RESTful API 설계 및 구현, "
            "관계형 DB 설계 및 최적화. "
            "우대: AWS/GCP 등 클라우드 플랫폼 경험, 대용량 트래픽 처리 경험"
        ),
        "required_skills": ["Python", "Java", "RESTful API", "MySQL", "클라우드"]
    },
    {
        "id": "job_002",
        "company": "네이버",
        "title": "서버 개발자 (신입)",
        "location": "경기 성남시 분당 (그린팩토리)",
        "work_type": "하이브리드 (주 3일 출근)",
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
        "location": "서울 강남구 역삼동",
        "work_type": "오피스 근무 (풀 재택 불가)",
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


def _filter_jobs_by_conditions(jobs: list, conditions: list) -> tuple:
    """
    job_conditions에 명확히 위배되는 공고를 사전 필터링한다.
    판단 불가한 경우는 기본 통과(benefit of doubt) 처리한다.
    반환: (통과한 공고 목록, 필터링된 공고 목록 with filter_reason)
    """
    if not conditions:
        return jobs, []

    llm = get_llm(temperature=0.1)

    conditions_text = "\n".join(
        f"- [{c['category']}] {c['condition']} (원문: \"{c['raw']}\")"
        for c in conditions
    )
    # 필터링에 필요한 공고 정보만 추출 (위치, 근무형태, JD)
    jobs_for_filter = json.dumps(
        [
            {
                "id": j["id"],
                "company": j["company"],
                "title": j["title"],
                "location": j.get("location", "정보 없음"),
                "work_type": j.get("work_type", "정보 없음"),
                "jd": j.get("jd", ""),
            }
            for j in jobs
        ],
        ensure_ascii=False,
        indent=2,
    )

    filter_prompt = f"""구직자의 필수 조건에 명확히 위배되는 공고를 식별하라.

[구직자 필수 조건]
{conditions_text}

[채용공고 목록]
{jobs_for_filter}

[판단 규칙]
- 공고의 location, work_type, jd 정보를 근거로 조건 위반 여부를 판단하라.
- 공고 정보만으로 충족 여부를 확실히 알 수 없는 경우: passes=true (기본 통과)
- 명확히 위반이 확인되는 경우에만 passes=false로 처리하라.
- 반드시 JSON 배열만 출력하라. 설명 텍스트, 마크다운 코드블록 일체 금지.

[출력 형식]
[
  {{"id": "job_001", "passes": true, "violation_reason": null}},
  {{"id": "job_002", "passes": false, "violation_reason": "위반된 조건과 이유를 한 문장으로"}}
]"""

    try:
        response = llm.invoke(filter_prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        filter_results = json.loads(raw.strip())

        result_map = {r["id"]: r for r in filter_results}
        passing, filtered = [], []
        for job in jobs:
            result = result_map.get(job["id"], {"passes": True, "violation_reason": None})
            if result.get("passes", True):
                passing.append(job)
            else:
                filtered.append({**job, "filter_reason": result.get("violation_reason", "")})

        print(f"[matching] 조건 필터링: 통과 {len(passing)}개 / 제외 {len(filtered)}개")
        return passing, filtered

    except (json.JSONDecodeError, Exception) as e:
        print(f"[matching] 조건 필터링 실패 (전체 통과 처리): {e}")
        return jobs, []


def matching_node(state: RookieState) -> dict:
    """
    UC2: 공고 매칭 에이전트 노드.
    1단계: job_conditions 위반 공고를 하드 필터링 (가중합 계산에서 완전 제외)
    2단계: 통과한 공고에 대해서만 매칭 점수 산출 후 match_score 내림차순 정렬
    """
    llm = get_llm(temperature=0.1)
    user_profile = state.get("user_profile", {})
    job_conditions = state.get("job_conditions", [])

    # 1단계: 필수 조건 위반 공고 사전 필터링
    passing_jobs, filtered_jobs = _filter_jobs_by_conditions(DUMMY_JOB_POSTINGS, job_conditions)

    # 조건이 너무 엄격해서 모든 공고가 필터링된 경우
    if not passing_jobs:
        cond_list = "\n".join(f"  • [{c['category']}] {c['condition']}" for c in job_conditions)
        no_result_msg = (
            f"설정하신 조건에 맞는 공고가 없어요 😢\n\n"
            f"[적용된 필터 조건]\n{cond_list}\n\n"
            "조건을 조정하시거나 범위를 넓혀볼까요?"
        )
        return {
            "matched_jobs": [],
            "current_step": "coaching",
            "messages": [AIMessage(content=no_result_msg)],
        }

    # 2단계: 통과한 공고에 대해서만 매칭 점수 산출
    jobs_text = json.dumps(passing_jobs, ensure_ascii=False, indent=2)
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

    # 결과 메시지 구성 (통과 공고 + 필터링된 공고 안내)
    result_lines = []
    if matched_jobs:
        result_lines.append("공고 매칭이 완료됐어요! 아래 공고들을 확인해 보세요:\n")
        for i, job in enumerate(matched_jobs, 1):
            score_pct = int(job.get("match_score", 0) * 100)
            result_lines.append(
                f"{i}. [{job['company']}] {job['title']} — 매칭률 {score_pct}%\n"
                f"   {job.get('jd_summary', '')}"
            )
    else:
        result_lines.append("죄송해요, 현재 조건에 맞는 공고를 찾지 못했어요.")

    # 필터링된 공고가 있으면 별도 안내
    if filtered_jobs:
        result_lines.append(f"\n\n[필터링된 공고 — 필수 조건 미충족으로 제외]")
        for job in filtered_jobs:
            result_lines.append(
                f"✕ [{job['company']}] {job['title']}\n"
                f"   사유: {job.get('filter_reason', '')}"
            )

    return {
        "matched_jobs": matched_jobs,
        "current_step": "coaching",
        "messages": [AIMessage(content="\n".join(result_lines))],
    }
