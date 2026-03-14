import os
from web_tools import WebSearchTool, CrawlPageTool


task="How many cups of water should adults drink per day?"
urls="https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/in-depth/water/art-20044256"
search_results = CrawlPageTool(
                f'http://{os.getenv("SERVER_HOST")}:{os.getenv("CRAWL_PAGE_PORT")}/crawl_page',
                api_key=os.getenv("SUMMARY_OPENAI_API_KEY"),
                api_url=os.getenv("SUMMARY_OPENAI_API_BASE_URL"),
                model=os.getenv("SUMMARY_MODEL"),
                task=task,
                urls=urls,
                history="",
            )

print(search_results)