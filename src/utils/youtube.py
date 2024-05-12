from googleapiclient.discovery import build


def make_youtube(api_key: str = 'AIzaSyC4CNo9YE6lzxiVwdHSU9-_Cq5AMhh-XRM'):
    return build('youtube', 'v3', developerKey=api_key)