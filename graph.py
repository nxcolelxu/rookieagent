from langgraph.graph import StateGraph, END
from state import RookieState
from nodes.profiling import profiling_node
from nodes.matching import matching_node
from nodes.coaching import coaching_node
from nodes.interview import interview_node


def build_graph():
    """
    루키 AI 메인 플로우 그래프를 구성하고 컴파일하여 반환한다.
    UC1(profiling) → UC2(matching) → UC3(coaching) → UC4(interview) → END
    """
    graph = StateGraph(RookieState)

    # 노드 등록
    graph.add_node("profiling", profiling_node)
    graph.add_node("matching", matching_node)
    graph.add_node("coaching", coaching_node)
    graph.add_node("interview", interview_node)

    # 진입점 설정
    graph.set_entry_point("profiling")

    # 엣지 연결 (선형 플로우)
    graph.add_edge("profiling", "matching")
    graph.add_edge("matching", "coaching")
    graph.add_edge("coaching", "interview")
    graph.add_edge("interview", END)

    # 그래프 컴파일 후 반환
    return graph.compile()
