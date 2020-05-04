import requests
import datetime
from dateutil.rrule import rrule, MONTHLY
import pyjq

nyt_apiKey= "####################################"

url = "https://api.nytimes.com/svc/archive/v1/2019/02.json?api-key=" + nyt_apiKey
req = requests.get.(url)
json_data = req.json()

#if you need to get copyright data
copyright = pyjq.all(".copyright", json_data)

# to get number of documents

num_docs = pyjq.all(".response .docs | length", json_data)[0]

jq_query = f'.response .docs [] | { {the_snippet: .snippet, the_headline: .headline .main, the_data: .pub_date, the_news_desk: .news_desk} }'
output = pyjq.all(jq_query, json_data)


# for extracting from daterange- month, year pair created
start_dt = datetime.date(2013,1,1)
end_dt = datetime.date(2020,4,1)
dates = [ (dt.year,dt.month) for dt in rrule(MONTHLY,dtstart= start_dt, until=end_dt) ]
all_output = []
for year, month in dates:
	time.sleep(20) # to slow down so query doesn't fail
	print(year,month)
	url = f'https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?&api-key={nyt_apiKey}'
	req = requests.get(url)
	json_data = req.json()

	length = pyjq.all(".response .docs | length", json_data)[0]
	print(f"For month {month} in {year} there were {length} articles")

	jq_query = f'.response .docs [] | { {the_snippet: .snippet, the_headline: .headline .main, the_data: .pub_date, the_news_desk: .news_desk} }'
	output = pyjq.all(jq_query, json_data)

	all_output.append([f"{year} {month:02}",output])