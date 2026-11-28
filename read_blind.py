import csv, json
f='outputs/evaluation/fusion/blind_spot_504.csv'
rows=list(csv.DictReader(open(f,encoding='utf-8')))
print('total rows',len(rows))
print('columns',list(rows[0].keys()))
corr=[r for r in rows if int(r['rule_correct'])==1]
print('rule_correct==1 count',len(corr))
import json
d=json.load(open('outputs/evaluation/fusion/blind_spot_disagreement.csv'))
print('disagreement keys',list(d.keys()) if isinstance(d,dict) else type(d))
