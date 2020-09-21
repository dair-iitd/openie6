import re
import difflib
import pickle
import argparse

def seq_in_seq(sub, full):
    return str(full)[1:-1].count(str(sub)[1:-1])

def starts_with(sub, full, index):
    return all(sub[i] == full[index + i] for i in range(0, len(sub)))

def main(input_file):

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines)):
        lines[i] = lines[i].strip().split('\t')
        lines[i][-1] = float(lines[i][-1])

    
    extractions = []
    for line in lines:
        extraction = {}
        assert len(line) == 3
        assert len(re.findall("<arg1>.*</arg1>", line[1])) == 1
        assert len(re.findall("<rel>.*</rel>", line[1])) == 1
        assert len(re.findall("<arg2>.*</arg2>", line[1])) == 1
        
        extraction["text"] = line[0] + " [unused1] [unused2] [unused3]"
        # extraction["text"] = line[0]
        extraction["tokens"] = extraction["text"].strip().split()
        extraction["tags"] = ['NONE']*len(extraction["tokens"])
        extraction["arg1"] = ' '.join(re.findall("<arg1>.*</arg1>", line[1])[0].strip('<arg1>').strip('</arg1>').strip().split())
        extraction["arg1_tokens"] = extraction["arg1"].strip().split()
        extraction["arg1_tagged"] = False
        extraction["rel"] = ' '.join(re.findall("<rel>.*</rel>", line[1])[0].strip('<rel>').strip('</rel>').strip().split())
        extraction["rel_tokens"] = extraction["rel"].strip().split()
        extraction["rel_tagged"] = False
        extraction["arg2"] = ' '.join(re.findall("<arg2>.*</arg2>", line[1])[0].strip('<arg2>').strip('</arg2>').strip().split())
        extraction["arg2_tokens"] = extraction["arg2"].strip().split()
        extraction["arg2_tagged"] = False
        extraction["confidence"] = line[2]
        extractions.append(extraction)
        
    sentences = set()
    for extraction in extractions:
        sentences.add(extraction["text"])

    print("Total number of sentences is : " + str(len(sentences)))
    print("Total number of extractions are : " + str(len(extractions)))

    count0, count1, count2, count3, count4, count5, count6 = 0, 0, 0, 0, 0, 0, 0

    for extraction in extractions:
        
        # fix arg2
        if extraction['arg2'] == '':
            count0 += 1
            extraction['arg2_tagged'] = True
        
        elif seq_in_seq(extraction["arg2_tokens"], extraction["tokens"]) == 1:
            # assert not extraction['arg2'] == ''
            matches = difflib.SequenceMatcher(None, extraction["arg2_tokens"], extraction["tokens"]).get_matching_blocks()
            if len(matches) == 2:
                count1 += 1
                assert matches[0].a == 0 and matches[0].size == matches[1].a and matches[1].size == 0
                extraction["arg2_tagged"] = True
                extraction["tags"][matches[0].b : matches[0].b + matches[0].size] = ["ARG2"] * matches[0].size
            else:
                print("exception ARG2")

        elif seq_in_seq(extraction["arg2_tokens"], extraction["tokens"]) == 0:
            matches = difflib.SequenceMatcher(None, extraction["arg2_tokens"], extraction["tokens"]).get_matching_blocks()
            if len(matches) > 2 and matches[0].a == 0 and all(matches[i].a == matches[i-1].a + matches[i-1].size for i in range(1, len(matches)-1)) and matches[-2].a + matches[-2].size == matches[-1].a:
                count2 += 1
                extraction["arg2_tagged"] = True
                for match in matches:
                    extraction["tags"][match.b : match.b + match.size] = ["ARG2"] * match.size
                
        #fix relation
        if seq_in_seq(extraction["rel_tokens"], extraction["tokens"]) == 1:
            count3 += 1
            matches = difflib.SequenceMatcher(None, extraction["rel_tokens"], extraction["tokens"]).get_matching_blocks()
            assert len(matches) == 2
            assert matches[0].a == 0 and matches[0].size == matches[1].a and matches[1].size == 0
            extraction["rel_tagged"] = True
            extraction["tags"][matches[0].b : matches[0].b + matches[0].size] = ["REL"] * matches[0].size
        
        #TODO - deal with closest match
        elif seq_in_seq(extraction["rel_tokens"], extraction["tokens"]) == 0:
            matches = difflib.SequenceMatcher(None, extraction["rel_tokens"], extraction["tokens"]).get_matching_blocks() 
            if len(matches) > 2 and matches[0].a == 0 and all(matches[i].a == matches[i-1].a + matches[i-1].size for i in range(1, len(matches)-1)) and matches[-2].a + matches[-2].size == matches[-1].a:
                count4 += 1
                extraction["rel_tagged"] = True
                for match in matches:
                    extraction["tags"][match.b : match.b + match.size] = ["REL"] * match.size
                    
                    
        # fix arg1
        if seq_in_seq(extraction["arg1_tokens"], extraction["tokens"]) == 1:
            matches = difflib.SequenceMatcher(None, extraction["arg1_tokens"], extraction["tokens"]).get_matching_blocks()
            if len(matches) == 2:
                count5 += 1
                assert matches[0].a == 0 and matches[0].size == matches[1].a and matches[1].size == 0
                extraction["arg1_tagged"] = True
                extraction["tags"][matches[0].b : matches[0].b + matches[0].size] = ["ARG1"] * matches[0].size
            else:
                print("exception ARG1")
            
        elif seq_in_seq(extraction["arg1_tokens"], extraction["tokens"]) == 0:
            matches = difflib.SequenceMatcher(None, extraction["arg1_tokens"], extraction["tokens"]).get_matching_blocks()
            if len(matches) > 2 and matches[0].a == 0 and all(matches[i].a == matches[i-1].a + matches[i-1].size for i in range(1, len(matches)-1)) and matches[-2].a + matches[-2].size == matches[-1].a:
                count6 += 1
                extraction["arg1_tagged"] = True
                for match in matches:
                    extraction["tags"][match.b : match.b + match.size] = ["ARG1"] * match.size
    print("Arg2 as empty string : " + str(count0))
    print("Arg2 exact match : " + str(count1))
    print("Arg2 piece-wise match : " + str(count2))
    print("Rel exact match : " + str(count3))
    print("Rel piece-wise match : " + str(count4))
    print("Arg1 exact match : " + str(count5))
    print("Arg1 piece-wise match : " + str(count6))

    # fix [is], [is] ... , [is] ... [of], [is] ... [from] relations
    count0, count1, count2, count3, count4, count5, count6 = 0, 0, 0, 0, 0, 0, 0

    for extraction in extractions:
        if not extraction["rel_tagged"] and len(extraction["rel_tokens"]) > 0:
            if extraction["rel"] == '[is]':
                count0 += 1
                extraction["rel_tagged"] = True
                assert extraction["tokens"][-3] == '[unused1]'
                extraction["tags"][-3] = 'REL'

            elif extraction["rel_tokens"][0] == '[is]' and extraction["rel_tokens"][-1] == '[of]':
                if len(extraction["rel_tokens"]) > 2 and seq_in_seq(extraction["rel_tokens"][1:-1], extraction["tokens"]) == 1:
                    matches = difflib.SequenceMatcher(None, extraction["rel_tokens"][1:-1], extraction["tokens"]).get_matching_blocks()
                    assert len(matches) == 2
                    assert matches[0].a == 0 and matches[0].size == matches[1].a and matches[1].size == 0
                    count1 += 1
                    extraction["rel_tagged"] = True
                    extraction["tags"][matches[0].b : matches[0].b + matches[0].size] = ["REL"] * matches[0].size
                    assert extraction["tokens"][-2] == '[unused2]'
                    extraction["tags"][-2] = 'REL'
                    
                elif len(extraction["rel_tokens"]) > 2 and seq_in_seq(extraction["rel_tokens"][1:-1], extraction["tokens"]) == 0:
                    matches = difflib.SequenceMatcher(None, extraction["rel_tokens"][1:-1], extraction["tokens"]).get_matching_blocks()
                    if len(matches) > 2 and matches[0].a == 0 and all(matches[i].a == matches[i-1].a + matches[i-1].size for i in range(1, len(matches)-1)) and matches[-2].a + matches[-2].size == matches[-1].a:
                        count2 += 1
                        extraction["rel_tagged"] = True
                        for match in matches:
                            extraction["tags"][match.b : match.b + match.size] = ["REL"] * match.size
                        assert extraction["tokens"][-2] == '[unused2]'
                        extraction["tags"][-2] = 'REL'

            elif extraction["rel_tokens"][0] == '[is]' and extraction["rel_tokens"][-1] == '[from]':
                if len(extraction["rel_tokens"]) > 2 and seq_in_seq(extraction["rel_tokens"][1:-1], extraction["tokens"]) == 1:
                    matches = difflib.SequenceMatcher(None, extraction["rel_tokens"][1:-1], extraction["tokens"]).get_matching_blocks()
                    assert len(matches) == 2
                    assert matches[0].a == 0 and matches[0].size == matches[1].a and matches[1].size == 0
                    count3 += 1
                    extraction["rel_tagged"] = True
                    extraction["tags"][matches[0].b : matches[0].b + matches[0].size] = ["REL"] * matches[0].size
                    assert extraction["tokens"][-1] == '[unused3]'
                    extraction["tags"][-1] = 'REL'
                    
                elif len(extraction["rel_tokens"]) > 2 and seq_in_seq(extraction["rel_tokens"][1:-1], extraction["tokens"]) == 0:
                    matches = difflib.SequenceMatcher(None, extraction["rel_tokens"][1:-1], extraction["tokens"]).get_matching_blocks()
                    if len(matches) > 2 and matches[0].a == 0 and all(matches[i].a == matches[i-1].a + matches[i-1].size for i in range(1, len(matches)-1)) and matches[-2].a + matches[-2].size == matches[-1].a:
                        count4 += 1
                        extraction["rel_tagged"] = True
                        for match in matches:
                            extraction["tags"][match.b : match.b + match.size] = ["REL"] * match.size
                        assert extraction["tokens"][-1] == '[unused3]'
                        extraction["tags"][-1] = 'REL'

            elif extraction["rel_tokens"][0] == '[is]' and len(extraction["rel_tokens"]) > 1:
                assert not extraction["rel_tokens"][-1].startswith('[')
                if seq_in_seq(extraction["rel_tokens"][1:], extraction["tokens"]) == 1:
                    matches = difflib.SequenceMatcher(None, extraction["rel_tokens"][1:], extraction["tokens"]).get_matching_blocks()
                    assert len(matches) == 2
                    assert matches[0].a == 0 and matches[0].size == matches[1].a and matches[1].size == 0
                    count5 += 1
                    extraction["rel_tagged"] = True
                    extraction["tags"][matches[0].b : matches[0].b + matches[0].size] = ["REL"] * matches[0].size
                    assert extraction["tokens"][-3] == '[unused1]'
                    extraction["tags"][-3] = 'REL'
                    
                elif len(extraction["rel_tokens"]) > 2 and seq_in_seq(extraction["rel_tokens"][1:], extraction["tokens"]) == 0:
                    matches = difflib.SequenceMatcher(None, extraction["rel_tokens"][1:-1], extraction["tokens"]).get_matching_blocks()
                    if len(matches) > 2 and matches[0].a == 0 and all(matches[i].a == matches[i-1].a + matches[i-1].size for i in range(1, len(matches)-1)) and matches[-2].a + matches[-2].size == matches[-1].a:
                        count6 += 1
                        extraction["rel_tagged"] = True
                        for match in matches:
                            extraction["tags"][match.b : match.b + match.size] = ["REL"] * match.size
                        assert extraction["tokens"][-3] == '[unused1]'
                        extraction["tags"][-3] = 'REL'

    print("Relations with [is], [of], [from] (exact and piece-wise) : " + str(count0 + count1 + count2 + count3 + count4 + count5 + count6))

    #fix multiple arg1
    count8 = 0
    for extraction in extractions:
        
        if extraction['rel_tagged'] and not extraction['arg1_tagged'] and seq_in_seq(extraction["arg1_tokens"], extraction["tokens"]) > 1:
            starting_indexes = [j for j in range(len(extraction["tokens"])) if starts_with(extraction["arg1_tokens"], extraction["tokens"], j)]
            assert len(starting_indexes) > 1
            
            min_dist = 10000000
            if 'REL' in extraction['tags']:
                rel_idx = extraction['tags'].index('REL')
                final_idx = -1

                for idx in starting_indexes:
                    dist = abs(rel_idx - idx)
                    if dist < min_dist:
                        min_dist = dist
                        final_idx = idx

                assert extraction['arg1_tokens'] == extraction['tokens'][final_idx : final_idx + len(extraction['arg1_tokens'])]
                count8 += 1
                extraction['arg1_tagged'] = True
                extraction['tags'][final_idx : final_idx + len(extraction['arg1_tokens'])] = ['ARG1'] * len(extraction['arg1_tokens'])
            else:
                print(extraction['text'])
    print("Multiple arg1 : " + str(count8))

    #fix multiple arg2
    count8 = 0
    for extraction in extractions:
        
        if extraction['rel_tagged'] and not extraction['arg2_tagged'] and seq_in_seq(extraction["arg2_tokens"], extraction["tokens"]) > 1:
            starting_indexes = [j for j in range(len(extraction["tokens"])) if starts_with(extraction["arg2_tokens"], extraction["tokens"], j)]
            assert len(starting_indexes) > 1
            
            min_dist = 10000000
            if 'REL' in extraction['tags']:
                rel_idx = extraction['tags'].index('REL')
                final_idx = -1

                for idx in starting_indexes:
                    dist = abs(rel_idx - idx)
                    if dist < min_dist:
                        min_dist = dist
                        final_idx = idx

                assert extraction['arg2_tokens'] == extraction['tokens'][final_idx : final_idx + len(extraction['arg2_tokens'])]
                count8 += 1
                extraction['arg2_tagged'] = True
                extraction['tags'][final_idx : final_idx + len(extraction['arg2_tokens'])] = ['ARG2'] * len(extraction['arg2_tokens'])
            else:
                print(extraction['text'])
    print("Multiple arg2 : " + str(count8))

    #fix multiple rel
    count9 = 0
    for extraction in extractions:
        if extraction['arg1_tagged'] and extraction['arg2_tagged'] and not extraction['rel_tagged'] and len(extraction["rel_tokens"]) > 0:
            
            rel_tokens= None
            if seq_in_seq(extraction["rel_tokens"], extraction["tokens"]) > 1:
                rel_tokens = extraction["rel_tokens"]
            elif extraction["rel_tokens"][0] == '[is]' and seq_in_seq(extraction["rel_tokens"][1:], extraction["tokens"]) > 1:
                rel_tokens = extraction["rel_tokens"][1:]
            elif extraction["rel_tokens"][0] == '[is]' and extraction["rel_tokens"][-1].startswith('[') and seq_in_seq(extraction["rel_tokens"][1:-1], extraction["tokens"]) > 1: 
                rel_tokens = extraction["rel_tokens"][1:-1]
                
            if rel_tokens:
                starting_indexes = [j for j in range(len(extraction["tokens"])) if starts_with(rel_tokens, extraction["tokens"], j)]
                assert len(starting_indexes) > 1

                min_dist = 10000000
                if 'ARG1' in extraction['tags'] and (extraction['arg2'] == '' or 'ARG2' in extraction['tags']):
                    arg1_idx = extraction['tags'].index('ARG1')
                    if extraction['arg2'] == '':
                        final_idx = -1
                        for idx in starting_indexes:
                            dist = abs(arg1_idx - idx)
                            if dist < min_dist:
                                min_dist = dist
                                final_idx = idx

                        assert rel_tokens == extraction['tokens'][final_idx : final_idx + len(rel_tokens)]
                        extraction['rel_tagged'] = True
                        count9 += 1
                        extraction['tags'][final_idx : final_idx + len(rel_tokens)] = ['REL'] * len(rel_tokens)

                    else:
                        arg2_idx = extraction['tags'].index('ARG2')
                        final_idx = -1
                        for idx in starting_indexes:
                            dist = abs(arg1_idx - idx) + abs(arg2_idx - idx)
                            if dist < min_dist:
                                min_dist = dist
                                final_idx = idx

                        assert rel_tokens == extraction['tokens'][final_idx : final_idx + len(rel_tokens)]
                        extraction['rel_tagged'] = True
                        count9 += 1
                        extraction['tags'][final_idx : final_idx + len(rel_tokens)] = ['REL'] * len(rel_tokens)
                
    print("Multiple rel : " + str(count9))

    
    
    count = 0
    for extraction in extractions:
        if extraction['arg2_tagged'] and extraction['rel_tagged'] and extraction['arg1_tagged']:
            if 'REL' in extraction['tags'] and 'ARG1' in extraction['tags']:
                if extraction['arg2'] == '' or 'ARG2' in extraction['tags']:
                    count += 1
    print("Total number of extractions parsed : " + str(count))

    with open(input_file + '.parsed', 'w') as f:
        current_tokens = []
        for extraction in extractions:
            if extraction['arg2_tagged'] and extraction['rel_tagged'] and extraction['arg1_tagged']:
                if 'REL' in extraction['tags'] and 'ARG1' in extraction['tags']:
                    if extraction['arg2'] == '' or 'ARG2' in extraction['tags']:
                        assert len(extraction['tokens']) == len(extraction['tags'])
                        if extraction['tokens'] != current_tokens:
                            current_tokens = extraction['tokens']
                            f.write(' '.join(extraction['tokens']))
                            f.write('\n')
                        f.write(' '.join(extraction['tags']))
                        f.write('\n')
        f.write('\n')

    print("Output written to file : " + input_file + '.parsed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="file path for gold in allennlp format", required=True)
    arguments = parser.parse_args()
    main(arguments.input)
