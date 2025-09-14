from dotenv import load_dotenv
import os
from asknews_sdk import AskNewsSDK
import requests

load_dotenv()

openrouter = os.getenv('OPENROUTER_API_KEY')


def test_basic_news():
    response = ask.news.search_news(
        query="EUR USD exchange rate forecast 2024",
        n_articles=5,
        return_type="string",
        method="kw"
    )
    print("Basic news search result:")
    print(response)
    print("\n" + "="*50 + "\n")

# Try minimal DeepNews call
def minimal_deep_research():
    try:
        response = ask.chat.get_deep_news(
            messages=[
                {
                    "role": "user",
                    "content": "What is the EUR USD exchange rate today?"
                }
            ],
            model="deepseek-basic",
            search_depth=2,
            max_depth=2,
            return_sources=False,
            filter_params=None
        )
        
        # response object matches the OpenAI SDK response object:
        print(response)
    except Exception as e:
        print(f"DeepNews error: {e}")


def check_openrouter_usage():
    url = "https://openrouter.ai/api/v1/key"

    headers = {"Authorization": f"Bearer {openrouter}"}

    response = requests.get(url, headers=headers)

    print(response.json())

check_openrouter_usage()


# minimal_deep_research()