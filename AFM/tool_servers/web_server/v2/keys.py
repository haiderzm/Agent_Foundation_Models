import os

serper_key_pool = [
    {
        "url": "https://google.serper.dev/search",
        "key": os.environ.get("WEB_SEARCH_SERPER_API_KEY")
    }
]

qwen_api_pool = [
    # {
    #     "url": "https://api.uniapi.vip/v1",
    #     "key": os.environ.get("UNI_API_KEY"),
    #     "model": "qwen2.5-72b-instruct",
    # },
    # {
    #     "url": "https://api.openai.com/v1",
    #     "key": os.environ.get("UNI_API_KEY"),
    #     "model": "gpt-4o-mini",
    # },
    {
        "url": "http://localhost:10000/v1",  
        "key": "empty",            
        "model": "Qwen3-30B-A3B-Instruct-2507-FP8",
    },
    # {
    #     "url": "https://api.deepinfra.com/v1/openai",
    #     "key": os.environ.get("UNI_API_KEY"),
    #     "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    # },
    # other qwen api provider ...
]


jina_api_pool = [
    {
        "key": os.environ.get("JINA_API_KEY")
    }
]


def get_serper_api() -> dict:
    """
    Returns a random Serper API from the pool.
    """
    import random
    return random.choice(serper_key_pool)

def get_qwen_api() -> dict:
    """
    Returns a random Qwen API from the pool.
    """
    import random
    return random.choice(qwen_api_pool)


def get_jina_api() -> dict:
    """
    Returns a random Jina API from the pool.
    """
    import random
    return random.choice(jina_api_pool)