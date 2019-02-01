import pickle
import numpy as np

attr2idx = pickle.load(open('../output/attr2idx.dict', 'rb'))
rare_user_col_index = attr2idx['UserId']['<RARE>']
rare_page_col_index = attr2idx['NaturalIDs']['<RARE>']


normal_user_col_indices = np.array([user_index
                                  for userid, user_index
                                  in attr2idx['UserId'].items()
                                  if userid !='<RARE>'])
normal_page_col_indices = np.array([page_index
                                  for pageid, page_index
                                  in attr2idx['NaturalIDs'].items()
                                  if pageid !='<RARE>'])
assert len(normal_user_col_indices) + 1 == len(attr2idx['UserId'])
assert len(normal_page_col_indices) + 1 == len(attr2idx['NaturalIDs'])

print(rare_user_col_index, rare_page_col_index)

times, events, sparse_features, sparse_headerbids = pickle.load(open('../output/TRAIN_SET.p', 'rb'))
sparse_features = sparse_features.tocsr()

def only_one_hot(row, rare_col_index, normal_col_indices):
    is_rare = row[rare_col_index]
    assert is_rare == 0 or is_rare == 1
    is_normal = sum(row[normal_col_indices])
    if is_normal + is_rare == 0:  # the one that we skipped (e.g., the most freq one)
        pass
    elif is_normal + is_rare == 1:  # correct
        pass
    else:
        print(is_rare, is_normal)

total = sparse_features.shape[0]
n = 0
for row in sparse_features:
    row = row.toarray()[0]

    n += 1
    if n % 100000 == 1:
        print('%d/%d (%f%%)' % (n, total, n/total*100))

    only_one_hot(row, rare_user_col_index, normal_user_col_indices)
    only_one_hot(row, rare_page_col_index, normal_page_col_indices)

print("Finish")