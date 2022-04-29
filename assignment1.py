import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head(10)

def date_sorter():

    global df
    # місяць [/|-] день [/|-] рік з 4ма знаками 
    # f.ex March-20-2009
    dates_extracted = df.str.extractall(r'(?P<origin>(?P<month>\d?\d)[/|-](?P<day>\d?\d)[/|-](?P<year>\d{4}))')
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
    # місяць [/|-] день (перше число 0-2, друге 0-9 + 31) [/|-] рік з 2ма знаками 
    # f.ex March-20-09
    dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>\d?\d)[/|-](?P<day>([0-2]?[0-9])|([3][01]))[/|-](?P<year>\d{2}))'))
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
    del dates_extracted[3]
    del dates_extracted[4]
    # день  місяць (всі літери, 3 і більше символів) [., ] рік з 4ма знаками 
    # f.ex 20-May-2009
    dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<day>\d?\d) ?(?P<month>[a-zA-Z]{3,})\.?,? (?P<year>\d{4}))'))
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])
    # місяць (всі літери, 3 і більше символів) [., ] день (закінчується на th|nd|st ) [,- ] рік з 4ма знаками 
    # f.ex May 20th, 2009
    dates_extracted = dates_extracted.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>[a-zA-Z]{3,})\.?-? ?(?P<day>\d\d?)(th|nd|st)?,?-? ?(?P<year>\d{4}))'))
    del dates_extracted[3]
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])

    # дані тільки місяць і рік
    # f.ex. May, 2009
    # день - перше число
    dates_without_day = df[index_left].str.extractall('(?P<origin>(?P<month>[A-Z][a-z]{2,}),?\.? (?P<year>\d{4}))')
    dates_without_day = dates_without_day.append(df[index_left].str.extractall(r'(?P<origin>(?P<month>\d\d?)/(?P<year>\d{4}))'))
    dates_without_day['day'] = 1
    dates_extracted = dates_extracted.append(dates_without_day)
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])

    # даний тільки рік
    # f.ex 2009
    dates_only_year = df[index_left].str.extractall(r'(?P<origin>(?P<year>\d{4}))')
    dates_only_year['day'] = 1
    dates_only_year['month'] = 1
    dates_extracted = dates_extracted.append(dates_only_year)
    index_left = ~df.index.isin([x[0] for x in dates_extracted.index])

    # визначення дня
    dates_extracted['day'] = dates_extracted['day'].apply(lambda x: str(x))
    
     # визначення місяця
    #
    dates_extracted['month'] = dates_extracted['month'].apply(lambda x: x[1:] if type(x) is str and x.startswith('0') else x)
    month_dict = dict({'Janaury': 1, 'January': 1, 'Jan': 1, 'Since': 1,
                       'February': 2, 'Feb': 2, 'Febuary': 2, 'Febraury': 2,
                       'March': 3, 'Mar': 3, 'Marc': 3, 'Marhc': 3,
                       'April': 4, 'Apr': 4, 'Aprli': 4, 'Apil': 4,
                       'May': 5, 'My': 5,
                       'June': 6, 'Jun': 6, 'Juen': 6, 
                       'July': 7, 'Jul': 7, 'Juyl': 7,
                       'August': 8, 'Aug': 8, 'Age': 8, 'Augus': 8, 'Auguts': 8,
                       'September': 9, 'Sep': 9, 'Setpember': 9, 'Setpembre': 9,
                       'October': 10, 'Oct': 10, 'Otcober': 10,
                       'November': 11, 'Nov': 11, 'Novebmer': 11,
                       'December': 12, 'Decemeber': 12, 'Dec': 12})
    dates_extracted.replace({"month": month_dict}, inplace=True)
    dates_extracted['month'] = dates_extracted['month'].apply(lambda x: str(x))
    
    # визначення року
    # f.ex. 20 -> 1920
    dates_extracted['year'] = dates_extracted['year'].apply(lambda x: '19' + x if len(x) == 2 else x)
    dates_extracted['year'] = dates_extracted['year'].apply(lambda x: str(x))

    # поєднання дати
    dates_extracted['date'] = dates_extracted['month'] + '/' + dates_extracted['day'] + '/' + dates_extracted['year']
    dates_extracted['date'] = pd.to_datetime(dates_extracted['date'])

    #сортування
    dates_extracted.sort_values(by='date', inplace=True)
    result = pd.Series(list(dates_extracted.index.labels[0]))
    
    return result
