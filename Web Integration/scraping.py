import wikipedia

def scrape_article(article_input, language='en'):
    try:
        wikipedia.set_lang(language)
        article_title = extract_article_title(article_input)
        page = wikipedia.page(article_title)
        scraped_content = page.content
    except wikipedia.exceptions.PageError:
        return None

    return scraped_content

def extract_article_title(article_input):
    if '/wiki/' in article_input:
        article_title = article_input.split('/wiki/')[1]
        if '#' in article_title:
            article_title = article_title.split('#')[0]
        return article_title
    else:
        return article_input