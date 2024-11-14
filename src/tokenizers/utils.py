import unicodedata

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers, replace all consecutive occurences of pairs with the new
    integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair(1, 2), idx=4 -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        # if i not at the last position AND the (i, i + 1) the matches the pair merge.
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
        
    return new_ids



def replace_control_characters(string: str):
    # dont print control characters
    chars = []
    for ch in string:
        if unicodedata.category(ch)[0] != 'C':
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    
    return "".join(chars)


def render_token(token: bytes):
    s = token.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s
