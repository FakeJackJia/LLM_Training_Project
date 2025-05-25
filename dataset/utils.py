import jsonlines

def jsonl_load(file):
    ret = []
    with jsonlines.open(file) as lines:
        for obj in lines:
            ret.append(obj)

    return ret