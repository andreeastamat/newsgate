import newspaper
import pandas as pd

data = ['https://www.nytimes.com/section/politics', 'https://www.wsj.com/', 'https://www.washingtonpost.com/',
        'https://empirenews.net/', 'https://newspunch.com/', 'http://beforeitsnews.com/']


def write():
    appended_data = []
    for eachWebsite in data:
        web = newspaper.build(eachWebsite, memoize_articles=False)
        print(web.size())
        try:
            for article in web.articles:
                article.download()
                article.parse()
                article.nlp()
                d = {'link': article.url,
                     'title': article.title,
                     'text': article.text,
                     'author': article.authors,
                     'date': article.publish_date,
                     'image': article.top_image}
                appended_data.append(d)
        except Exception as e:
            print(e)
            continue

    df = pd.DataFrame(appended_data)
    print(df.shape)
    df.to_csv(r'dataset_news.csv', sep='/', index=False)


write()