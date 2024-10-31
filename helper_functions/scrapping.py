from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool

@tool
def scrape_web_pages(urls):
    '''
    Use this tool to scrape web pages from link.

    Args:
            urls (list): A list of strings where each string is a link to some article.

    
    Returns:
            list: A list of strings where each string is a scrapped content of the corresponding links of input list.
    '''
    loader = WebBaseLoader(urls)  
    documents = loader.load()      
    return [doc.page_content for doc in documents]  