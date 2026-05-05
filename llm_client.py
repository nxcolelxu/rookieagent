from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    SKT A.X 4.0 LLM 클라이언트를 반환한다.
    모든 노드에서 LLM이 필요할 때 이 함수를 호출한다.
    """
    return ChatOpenAI(
        model=os.getenv("SKT_AX_MODEL_NAME", "ax-4"),
        api_key=os.getenv("SKT_AX_API_KEY"),
        base_url=os.getenv("SKT_AX_BASE_URL"),
        temperature=temperature,
    )
