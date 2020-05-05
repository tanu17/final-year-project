# -*- coding: cp1252 -*-

from os import walk
import pandas as pd

mypath_short="D:/NTU/Year 4/Semester 2/FYP/Analysis of news data/Context article/Shortened"
mypath_long= "D:/NTU/Year 4/Semester 2/FYP/Analysis of news data/Context article/Full length"

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def quotation_exractor(context):
	#list_of_letters=context.split()
	q_list=[]
	opening_list = find(str(context), "“" )
	closing_list = find(str(context), "”")

	if len(closing_list)==len(opening_list):
		for i in range(len(opening_list)):
			q_list.append(context[opening_list[i]:closing_list[i]])
	else:
		if len(opening_list)-len(closing_list)==1:
			for i in range(len(opening_list)-1):
				q_list.append(context[opening_list[i]:closing_list[i]])
		else:
			pass
	return(q_list)
important_words=["belt", "road", "bri", "debt", "trap", "xi", "beijing"]



df = pd.read_csv('C:/Users/User/Desktop/China_withContext.csv')
context_of_all_files=df['Context']
print(len(context_of_all_files))
list_ofListofCountries= df.Countries


def clean_data(context, country_list):
    relavent_context = []
    context= str(context)
    lines = context.split("\n")

    yes=0
    for line in lines:
        i =lines.index(line)
        for word in line:
            try:
                if word.lower() in country_list:
                    try:
                        relavent_context += [lines[i - 2], lines[i - 1], line, lines[i + 1], lines[i + 2]]
                    except:
                        try:
                            relavent_context += [lines[i - 2], lines[i - 1], line]
                        except:
                            relavent_context +=[line]
                    continue
            except:
                continue
        for elemnt in important_words:
            if elemnt in line:
                try:
                    relavent_context += [lines[i - 2], lines[i - 1], line, lines[i + 1], lines[i + 2]]
                except:
                    try:
                        relavent_context += [lines[i - 2], lines[i - 1], line]
                    except:
                        relavent_context +=[line]
                continue

    relavent_context = list(set(relavent_context))
    return relavent_context

f_short = []
f_long = []
for (dirpath, dirnames, filenames) in walk(mypath_short):
    f_short.extend(filenames)
for (dirpath, dirnames, filenames) in walk(mypath_long):
    f_long.extend(filenames)
f_short.sort()
f_long.sort()


for i in range(1,512):
    str_new=str(i)+".txt"
    print(str_new)

    if str_new in f_short:
        file_context= open("D:/NTU/Year 4/Semester 2/Shimmy shimmy ya/Analysis of news data/Context article/Shortened/"+str_new,"r")
        file_context = file_context.read()
        q_list = quotation_exractor(file_context)
        with open("D:/NTU/Year 4/Semester 2/Shimmy shimmy ya/Analysis of news data/Context article/Quotes/"+str(i)+"_quote.txt", 'w') as output:
            output.writelines(q_list)
            output.flush()
            output.close()

    elif (str_new in f_long):
    	file_context= open("D:/NTU/Year 4/Semester 2/FYP/Analysis of news data/Context article/Full length/"+str_new,"r")
    	file_context = file_context.read()
    	q_list = quotation_exractor(file_context)
    	with open("D:/NTU/Year 4/Semester 2/FYP/Analysis of news data/Context article/Quotes/"+str(i)+"_quote.txt", 'w') as output:
    		output.writelines(q_list)
    		output.flush()
    		output.close()
    	cleaned = clean_data(file_context, list_ofListofCountries[i-1])
    	
    	with open("D:/NTU/Year 4/Semester 2/FYP/Analysis of news data/Context article/Shortened/"+str(i)+".txt", 'w') as output1:
    		output1.writelines(cleaned)
    		output1.flush()
    		output1.close()
    else:
        file_context= context_of_all_files[i-1]
        if file_context == "":
            continue
        q_list = quotation_exractor(file_context)
        with open("D:/NTU/Year 4/Semester 2/FYP/Analysis of news data/Context article/Quotes/"+str(i)+"_quote.txt", 'w') as output:
                                        output.writelines(q_list)
                                        output.flush()
                                        output.close()
        cleaned = clean_data(file_context, list_ofListofCountries[i-1])

        with open("D:/NTU/Year 4/Semester 2/FYP/Analysis of news data/Context article/Shortened/"+str(i)+".txt", 'w') as output1:
                                        output1.writelines(cleaned)
                                        output1.flush()
                                        output1.close()
