import pickle

MIN_OCCURRENCE = 5
counter = pickle.load(
    open("../counter.dict", "rb"))  # {Attribute1:Counter<features>, Attribute2:Counter<features>, ...}
attr2idx = pickle.load(
    open("../attr2idx.dict", "rb"))  # {Attribute1: dict(feat1:i, ...), Attribute2: dict(feat1:i, ...), ...}

for attr in counter:
    print("ATTRIBUTE %s" % attr)
    for feat in counter[attr]:
        if feat in attr2idx[attr]:
            continue
        else:
            if counter[attr][feat] < MIN_OCCURRENCE:
                if attr not in ['NaturalIDs', 'UserId', 'RefererURL']:
                    print("    FEATURE %s is rare" % feat)
            else:
                print("    FEATURE %s was ignored" % feat)
