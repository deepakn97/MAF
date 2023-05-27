import json



def main():
    fp = '3shot_entailment.txt'
    with open(fp, 'r') as f:
        raw = f.read()
    raw = raw.split('USER:')[1:]
    print(raw)
    raw = [{
        'question': x.split('ASSISTANT:')[0].strip(),
        'solution': x.split('ASSISTANT:')[1].strip()
    } for x in raw if 'ASSISTANT:' in x]
    with open('3shot_entailment.json', 'w') as f:
        json.dump(raw, f, indent=2)
if __name__ == '__main__':
    main()