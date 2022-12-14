import csv
import sys
import glob
import os
import json

input_folder = sys.argv[1]
csv_output = sys.argv[2]

all_output_folder = glob.glob(os.path.join(input_folder,'*'))
print(all_output_folder)

f_csv = open(csv_output, 'w', newline='')
spamwriter = csv.writer(f_csv, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(['Sentence','type of impact', 'URL', 'date', 'pest', 'disease', 'host'])


for each_sub_folder in all_output_folder:
    full_json_path = os.path.join(each_sub_folder,'summary.json')
    with open(full_json_path, 'r') as fp:
        data = json.load(fp)
    for each_data in data:
        if len(each_data['items']) > 0:
            print('111')
            for each_item in each_data['items']:
                sentence = each_item['Orign_Sentence']
                type_of_impact = each_item['Type_of_impact']
                url = each_data['url']
                date = each_data['date_of_the_artical']
                pest = each_item['Pest']['Scientific_name']
                disease = each_item['Disease']['Scientific_name']
                host = each_item['Host']['Scientific_name']
                spamwriter.writerow([sentence, type_of_impact, url, date, pest, disease, host])

f_csv.close()






