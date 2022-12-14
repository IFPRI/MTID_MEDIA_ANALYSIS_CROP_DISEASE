import sys
import json

input_json = sys.argv[1]

with open(input_json, 'r') as fp:
    data = json.load(fp)

#print('Sentence\tHost_sci\tOri_country\tDisease_sci\tPest_sci\tImpact_country\ttype_of_impact\turl')
for each_artical in data:
    url = each_artical['url']
    for each_item in each_artical['items']:
        sentence = each_item['Orign_Sentence'].replace('\n', ' ')
        type_of_impact = each_item['Type_of_impact']
        Origin_country = str(each_item['Origin_country'])
        Disease = each_item['Disease']
        Disease_sci = str(Disease['Scientific_name'])
        Pest = each_item['Pest']
        Pest_sci = str(Pest['Scientific_name'])
        Host = each_item['Host']
        Host_sci = str(Host['Scientific_name'])
        Impacted_area = each_item['Impacted_area']
        Impacted_area_country = str(Impacted_area['Country'])

        print(sentence+'\t'+Host_sci+'\t'+Origin_country+'\t'+Disease_sci+'\t'+Pest_sci+'\t'+Impacted_area_country+'\t'+type_of_impact+'\t'+url)
