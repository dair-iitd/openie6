import ipdb
global mapping, unk_mapping

def init_mapping():
    global mapping, unk_mapping
    mapping = dict()
    unk_mapping = dict()
    unk_num = 1
    for string in ['<arg1>', '</arg1>', '<rel>', '</rel>', '<arg2>', '</arg2>', 'SENT', 'PRED', '@COPY@', 'EOE']:
        unk_str = '[unused'+str(unk_num)+']'
        mapping[string] = unk_str
        unk_mapping[unk_str] = string

        unk_num += 1

    print(mapping)

init_mapping()

def init_globals():
    return "[CLS]", "[SEP]"

def replace_strings(string_):
    if string_ is not None:
        for key in mapping:
            string_ = string_.replace(key, mapping[key])

    return string_
    
