##########################
### Utilities Function ###
##########################

# one-hot encoding (only allowable set)
def one_of_k_encoding(x, allowable_set):
    # Maps inputs only in the allowable set 
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set {1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

# one-hot encoding (unknown to last element)
def one_of_k_encoding_unk(x, allowable_set):
    # Maps inputs not in the allowable set to the last element
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# one-hot encoding (unknown all zeros)
def one_of_k_encoding_none(x, allowable_set):
    # Maps inputs not in the allowable set to zero list
    if x not in allowable_set:
        x = [0 for i in range(len(allowable_set))]
    return list(map(lambda s: x == s, allowable_set))