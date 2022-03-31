from qwikidata.sparql import (get_subclasses_of_item, return_sparql_query_results)
import json
import time
import re


def flatten_list(nested):
    if isinstance(nested, list):
        for sublist in nested:
            for item in flatten_list(sublist):
                yield item
    else:
        yield nested


def type_query(entity_id):
    sparql_query = """
    SELECT ?item ?itemLabel
    WHERE
    {
        wd:%s wdt:P31 ?item.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    """ % entity_id

    res = return_sparql_query_results(sparql_query)
    results = res['results']['bindings']
    type_raw = [item['itemLabel'] for item in results]
    type_list = [item['value'] for item in type_raw]
    return type_list


file_url = "../data/wikipeople/n-ary_train.json"
file = open(file_url, 'r', encoding='utf-8')
triple_list = []
# i=0
for line in file.readlines():  # read the json file line by line
    # i = i + 1
    dic = json.loads(line)
    triple_list.append(dic)
    # if i>10:
    #     break

file.close()

nary_list = [item for item in triple_list if item['N'] > 2]
kv_train_type_list = []
i = 0
for item in nary_list:
    entity_list = list(flatten_list(list(item.values())))[2:-1]
    for entity_id in entity_list:
        time.sleep(1)
        type_list = type_query(entity_id)
        entity_dup = [entity_id] * len(type_list)
        entity_zip = list(zip(entity_dup, type_list))
        kv_train_type_list.extend(entity_zip)

    i = i + 1
    if (i % 500) == 0:
        print("process %d facts" % i)
        time.sleep(60)


kv_type_list = kv_train_type_list + kv_test_type_list + kv_valid_type_list
kv_type_list = list(set([tuple(t) for t in kv_type_list]))
kv_type_list = sorted(kv_type_list, key=(lambda x: x[0]))


file_output = open("../data/wikipeople/hyper-relation_value2type.txt", "w")
for line in kv_type_list:
    file_output.write('/value/' + line[0] + ' ' + '/type/' + line[1] + '\n')

file_output.close()
# nary_list = [item for item in triple_list if item['N'] > 2]  # find the triples whose arity>2


#
#
# nary_nonliteral = []
# for item in nary_list:     # remove triples with literals
#     item_c = item.copy()
#     item_c.pop('N')
#     value_list = list(flatten_list(list(item_c.values())))
#     flag = 0
#     for k in value_list:
#         if re.match('Q', k) == None:
#             flag = 1
#             break
#     if flag == 0:
#         nary_nonliteral.append(item)


#
# print(res)
# print(results)
# print(type_raw)
# print(type_list)
