from indictrans import Transliterator
from langchain_core.tools import tool

@tool
def transliterate_hindi_to_english(hindi_text):
    '''
    Use this tool to transliterate hindi to english.
    Args:
            query (str): the text to be transliterated from hindi to english.

    
    Returns:
            str: translated text.
    '''
    trn = Transliterator(source='hin', target='eng', build_lookup=True)
    english_text = trn.transform(hindi_text)
    return english_text

# # Example usage
# hindi_text = "यह एक उदाहरण है"
# english_text = transliterate_hindi_to_english(hindi_text)
# print("Transliterated Text:", english_text)
