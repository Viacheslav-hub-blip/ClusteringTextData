from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.embeddings import OpenAIEmbeddings

model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    #api_key="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",
    api_key="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",
    model="z-ai/glm-5",
    #model="nvidia/nemotron-3-super-120b-a12b",
    # model="google/gemini-2.5-flash",
    # google/gemini-3-pro-preview
    # google/gemini-2.5-flash
    # model="kwaipilot/kat-coder-pro:free",
    temperature=0.1
)

embeddings = OpenAIEmbeddings(
    # 1. Меняем базовый URL на OpenRouter
    base_url="https://openrouter.ai/api/v1",

    # 2. Передаем ключ OpenRouter
    api_key="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",

    # 3. Указываем модель (OpenRouter требует указывать провайдера, например 'openai/')
    model="openai/text-embedding-3-small",

    # Опционально: отключаем проверку SSL, если возникают странные ошибки сети
    # check_embeddings=True
)


def get_answer(prompt: str, model, prompt_params: dict = None) -> str:
    if prompt_params is None:
        prompt_params = {}
    chain = ChatPromptTemplate.from_template(prompt) | model | StrOutputParser()
    return chain.invoke(prompt_params)


if __name__ == "__main__":
    print(model.invoke("hi"))
