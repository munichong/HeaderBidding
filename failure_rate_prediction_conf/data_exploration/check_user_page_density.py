import pickle

from pymongo import MongoClient

USER_MAX_BIN = 200
PAGE_MAX_BIN = 500

client = MongoClient()
n_total_imps = client['Header_Bidding']['NetworkImpressions'].find().count() + \
               client['Header_Bidding']['NetworkBackfillImpressions'].find().count()

counter = pickle.load(
    open("../output/counter.dict", "rb"))  # {Attribute1:Counter<features>, Attribute2:Counter<features>, ...}

user_counter = counter['UserId']
page_counter = counter['NaturalIDs']
print(user_counter.most_common(100))
print(page_counter.most_common(100))

user_bin = [0] * USER_MAX_BIN
for u, c in user_counter.most_common():
    if c <= USER_MAX_BIN - 1:
        user_bin[c - 1] += 1
    else:
        user_bin[-1] += 1
for i, c in enumerate(user_bin[:-1]):
    print("%d users have %d impressions" % (c, i + 1))
print("%d users have >=%d (%f%%) impressions" % (user_bin[-1], len(user_bin), user_bin[-1] / sum(user_bin) * 100))
print()

page_bin = [0] * PAGE_MAX_BIN
for p, c in page_counter.most_common():
    if c <= PAGE_MAX_BIN - 1:
        page_bin[c - 1] += 1
    else:
        page_bin[-1] += 1
for i, c in enumerate(page_bin[:-1]):
    print("%d pages have %d impressions" % (c, i + 1))
print("%d pages have >=%d (%f%%) impressions" % (page_bin[-1], len(page_bin), page_bin[-1] / sum(page_bin) * 100))
print()

print("%d users" % len(user_counter))
print("%d pages" % len(page_counter))
print("%d impressions" % n_total_imps)
print("Density = %f%%" % (n_total_imps / (len(user_counter) * len(page_counter)) * 100))
