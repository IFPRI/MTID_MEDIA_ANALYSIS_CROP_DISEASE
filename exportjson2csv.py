import csv
import sys
import glob
import os
import json
import nltk


def compare_title(title, processed_title):
    similar_found = False
    similar_title = ''
    for each_processed_title in processed_title:
        token_level_score = nltk.edit_distance(title.lower().strip().split(), each_processed_title.lower().strip().split())
        if token_level_score < 3:
            similar_found = True
            similar_title = each_processed_title
            break
    return similar_found, similar_title

input_folder = sys.argv[1]
csv_output = sys.argv[2]

all_output_folder = glob.glob(os.path.join(input_folder,'*'))
print(all_output_folder)

f_csv = open(csv_output, 'w', newline='')
spamwriter = csv.writer(f_csv, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(['Sentence','type of impact', 'URL', 'date', 'pest', 'disease', 'host', 'Impacted_area'])

processed_title = []


for each_sub_folder in all_output_folder:
    full_json_path = os.path.join(each_sub_folder,'summary.json')
    with open(full_json_path, 'r') as fp:
        data = json.load(fp)
    for each_data in data:
        title = each_data['title']
        similar_found = False
        if title:
            similar_found, similar_title = compare_title(title, processed_title)
            #if similar_found:
            #    print(each_data['title'], similar_title)
            processed_title.append(each_data['title'])

        if len(each_data['items']) > 0 and not similar_found:
            print('processing: ', title)
            for each_item in each_data['items']:
                sentence = each_item['Orign_Sentence']
                type_of_impact = each_item['Type_of_impact']
                url = each_data['url']
                date = each_data['date_of_the_artical']
                pest = each_item['Pest']['Scientific_name']
                disease = each_item['Disease']['Scientific_name']
                host = each_item['Host']['Scientific_name']
                impacted_area = each_item['Impacted_area']['Country']
                spamwriter.writerow([sentence, type_of_impact, url, date, pest, disease, host, impacted_area])

f_csv.close()






