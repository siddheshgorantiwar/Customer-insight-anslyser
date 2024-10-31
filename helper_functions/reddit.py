import praw
import os
from dotenv import load_dotenv
from langchain_core.tools import tool


# Load environment variables from .env file
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    user_agent=os.getenv('USER_AGENT')
)

@tool
def get_reddit(query, max_results):
    '''
    Use this tool to get links from reddit.

    Args:
            query (str): The search query for finding reddit posts.
            max_results (int): The maximum number of links to retrieve.

    
    Returns:
            list: A list of strings where each string is a link to some article.
    '''
    urls=[]
    subreddit = reddit.subreddit('all')  
    for submission in subreddit.top(limit=max_results):
        urls.append(submission.url)
    
    return urls

########## SAMPLE CALL #############
# links=get_reddit("iphone 16", 5)
# print(links)