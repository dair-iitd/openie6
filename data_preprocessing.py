import re
import ipdb
import difflib
import pickle
import argparse


def seq_in_seq(sub, full):
    return str(full)[1:-1].count(str(sub)[1:-1])


def starts_with(sub, full, index):
    return all(sub[i] == full[index + i] for i in range(0, len(sub)))


def label_arg2(extraction):

    def label_extraction(matches):
        if len(matches) == 2:
            assert matches[0].a == 0 and matches[0].size == matches[1].a and matches[1].size == 0
            extraction["arg2_tagged"] = True
            extraction["tags"][matches[0].b: matches[0].b +
                               matches[0].size] = ["ARG2"] * matches[0].size

    if extraction['arg2'] == '' and len(extraction["args_tokens"]) == 0 and len(extraction["loc_args_tokens"]) == 0 and len(extraction["time_args_tokens"]) == 0:
        extraction['arg2_tagged'] = True

    elif seq_in_seq(extraction["arg2_tokens"] + extraction["args_tokens"] + extraction["loc_args_tokens"] + extraction["time_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["arg2_tokens"] + extraction["args_tokens"] + extraction["loc_args_tokens"] + extraction["time_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)
            
    elif seq_in_seq(extraction["arg2_tokens"] + extraction["args_tokens"] + extraction["time_args_tokens"] + extraction["loc_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["arg2_tokens"] + extraction["args_tokens"] + extraction["time_args_tokens"] + extraction["loc_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)
            
    elif seq_in_seq(extraction["arg2_tokens"] + extraction["args_tokens"] + extraction["time_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["arg2_tokens"] + extraction["args_tokens"] + extraction["time_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)
            
    elif seq_in_seq(extraction["arg2_tokens"] + extraction["args_tokens"] + extraction["loc_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["arg2_tokens"] + extraction["args_tokens"] + extraction["loc_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)
            
    elif seq_in_seq(extraction["arg2_tokens"] + extraction["time_args_tokens"] + extraction["loc_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["arg2_tokens"] + extraction["time_args_tokens"] + extraction["loc_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)
            
    elif seq_in_seq(extraction["arg2_tokens"] + extraction["loc_args_tokens"] + extraction["time_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["arg2_tokens"] + extraction["loc_args_tokens"] + extraction["time_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)
            
    elif seq_in_seq(extraction["arg2_tokens"] + extraction["time_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["arg2_tokens"] + extraction["time_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)
            
    elif seq_in_seq(extraction["arg2_tokens"] + extraction["loc_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["arg2_tokens"] + extraction["loc_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)
            
    elif seq_in_seq(extraction["time_args_tokens"] + extraction["loc_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["time_args_tokens"] + extraction["loc_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)
            
    elif seq_in_seq(extraction["loc_args_tokens"] + extraction["time_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["loc_args_tokens"] + extraction["time_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)
            
    elif seq_in_seq(extraction["loc_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["loc_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)
            
    elif seq_in_seq(extraction["time_args_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(None, extraction["time_args_tokens"], extraction["tokens"]).get_matching_blocks()
        label_extraction(matches)


def label_arg(extraction, arg):

    if seq_in_seq(extraction[arg + "_tokens"], extraction["tokens"]) == 1:
        matches = difflib.SequenceMatcher(
            None, extraction[arg + "_tokens"], extraction["tokens"]).get_matching_blocks()
        assert len(matches) == 2
        assert matches[0].a == 0 and matches[0].size == matches[1].a and matches[1].size == 0
        extraction[arg + "_tagged"] = True
        extraction["tags"][matches[0].b : matches[0].b + matches[0].size] = [arg.upper()] * matches[0].size
    
    elif seq_in_seq(extraction[arg + "_tokens"], extraction["tokens"]) == 0:
        matches = difflib.SequenceMatcher(
            None, extraction[arg + "_tokens"], extraction["tokens"]).get_matching_blocks()
        if len(matches) > 2 and matches[0].a == 0 and all(matches[i].a == matches[i-1].a + matches[i-1].size for i in range(1, len(matches)-1)) and matches[-2].a + matches[-2].size == matches[-1].a:
            extraction[arg + "_tagged"] = True
            for match in matches:
                extraction["tags"][match.b : match.b + match.size] = [arg.upper()] * match.size


def label_is_of_relations(extractions):

    for extraction in extractions:
        if not extraction["rel_tagged"] and len(extraction["rel_tokens"]) > 0:
            if extraction["rel"] == '[is]':
                extraction["rel_tagged"] = True
                assert extraction["tokens"][-3] == '[unused1]'
                extraction["tags"][-3] = 'REL'

            elif extraction["rel_tokens"][0] == '[is]' and extraction["rel_tokens"][-1] == '[of]':
                if len(extraction["rel_tokens"]) > 2 and seq_in_seq(extraction["rel_tokens"][1:-1], extraction["tokens"]) == 1:
                    matches = difflib.SequenceMatcher(
                        None, extraction["rel_tokens"][1:-1], extraction["tokens"]).get_matching_blocks()
                    assert len(matches) == 2
                    assert matches[0].a == 0 and matches[0].size == matches[1].a and matches[1].size == 0
                    extraction["rel_tagged"] = True
                    extraction["tags"][matches[0].b: matches[0].b +
                                    matches[0].size] = ["REL"] * matches[0].size
                    assert extraction["tokens"][-2] == '[unused2]'
                    extraction["tags"][-2] = 'REL'

                elif len(extraction["rel_tokens"]) > 2 and seq_in_seq(extraction["rel_tokens"][1:-1], extraction["tokens"]) == 0:
                    matches = difflib.SequenceMatcher(
                        None, extraction["rel_tokens"][1:-1], extraction["tokens"]).get_matching_blocks()
                    if len(matches) > 2 and matches[0].a == 0 and all(matches[i].a == matches[i-1].a + matches[i-1].size for i in range(1, len(matches)-1)) and matches[-2].a + matches[-2].size == matches[-1].a:
                        extraction["rel_tagged"] = True
                        for match in matches:
                            extraction["tags"][match.b: match.b +
                                            match.size] = ["REL"] * match.size
                        assert extraction["tokens"][-2] == '[unused2]'
                        extraction["tags"][-2] = 'REL'

            elif extraction["rel_tokens"][0] == '[is]' and extraction["rel_tokens"][-1] == '[from]':
                if len(extraction["rel_tokens"]) > 2 and seq_in_seq(extraction["rel_tokens"][1:-1], extraction["tokens"]) == 1:
                    matches = difflib.SequenceMatcher(
                        None, extraction["rel_tokens"][1:-1], extraction["tokens"]).get_matching_blocks()
                    assert len(matches) == 2
                    assert matches[0].a == 0 and matches[0].size == matches[1].a and matches[1].size == 0
                    extraction["rel_tagged"] = True
                    extraction["tags"][matches[0].b: matches[0].b +
                                    matches[0].size] = ["REL"] * matches[0].size
                    assert extraction["tokens"][-1] == '[unused3]'
                    extraction["tags"][-1] = 'REL'

                elif len(extraction["rel_tokens"]) > 2 and seq_in_seq(extraction["rel_tokens"][1:-1], extraction["tokens"]) == 0:
                    matches = difflib.SequenceMatcher(
                        None, extraction["rel_tokens"][1:-1], extraction["tokens"]).get_matching_blocks()
                    if len(matches) > 2 and matches[0].a == 0 and all(matches[i].a == matches[i-1].a + matches[i-1].size for i in range(1, len(matches)-1)) and matches[-2].a + matches[-2].size == matches[-1].a:
                        extraction["rel_tagged"] = True
                        for match in matches:
                            extraction["tags"][match.b: match.b +
                                            match.size] = ["REL"] * match.size
                        assert extraction["tokens"][-1] == '[unused3]'
                        extraction["tags"][-1] = 'REL'

            elif extraction["rel_tokens"][0] == '[is]' and len(extraction["rel_tokens"]) > 1:
                assert not extraction["rel_tokens"][-1].startswith('[')
                if seq_in_seq(extraction["rel_tokens"][1:], extraction["tokens"]) == 1:
                    matches = difflib.SequenceMatcher(
                        None, extraction["rel_tokens"][1:], extraction["tokens"]).get_matching_blocks()
                    assert len(matches) == 2
                    assert matches[0].a == 0 and matches[0].size == matches[1].a and matches[1].size == 0
                    extraction["rel_tagged"] = True
                    extraction["tags"][matches[0].b: matches[0].b +
                                    matches[0].size] = ["REL"] * matches[0].size
                    assert extraction["tokens"][-3] == '[unused1]'
                    extraction["tags"][-3] = 'REL'

                elif len(extraction["rel_tokens"]) > 2 and seq_in_seq(extraction["rel_tokens"][1:], extraction["tokens"]) == 0:
                    matches = difflib.SequenceMatcher(
                        None, extraction["rel_tokens"][1:-1], extraction["tokens"]).get_matching_blocks()
                    if len(matches) > 2 and matches[0].a == 0 and all(matches[i].a == matches[i-1].a + matches[i-1].size for i in range(1, len(matches)-1)) and matches[-2].a + matches[-2].size == matches[-1].a:
                        extraction["rel_tagged"] = True
                        for match in matches:
                            extraction["tags"][match.b: match.b +
                                            match.size] = ["REL"] * match.size
                        assert extraction["tokens"][-3] == '[unused1]'
                        extraction["tags"][-3] = 'REL'


def label_multiple_arg1(extractions):

    for extraction in extractions:
        if extraction['rel_tagged'] and not extraction['arg1_tagged'] and seq_in_seq(extraction["arg1_tokens"], extraction["tokens"]) > 1:
            starting_indexes = [j for j in range(len(extraction["tokens"])) if starts_with(
                extraction["arg1_tokens"], extraction["tokens"], j)]
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

                assert extraction['arg1_tokens'] == extraction['tokens'][final_idx: final_idx + len(
                    extraction['arg1_tokens'])]
                extraction['arg1_tagged'] = True
                extraction['tags'][final_idx: final_idx + len(extraction['arg1_tokens'])] = [
                    'ARG1'] * len(extraction['arg1_tokens'])
            else:
                assert False


def label_multiple_rel(extractions):

    for extraction in extractions:

        if extraction['arg1_tagged'] and extraction['arg2_tagged'] and not extraction['rel_tagged'] and len(extraction["rel_tokens"]) > 0:
            rel_tokens = None
            if seq_in_seq(extraction["rel_tokens"], extraction["tokens"]) > 1:
                rel_tokens = extraction["rel_tokens"]
            elif extraction["rel_tokens"][0] == '[is]' and seq_in_seq(extraction["rel_tokens"][1:], extraction["tokens"]) > 1:
                rel_tokens = extraction["rel_tokens"][1:]
            elif extraction["rel_tokens"][0] == '[is]' and extraction["rel_tokens"][-1].startswith('[') and seq_in_seq(extraction["rel_tokens"][1:-1], extraction["tokens"]) > 1:
                rel_tokens = extraction["rel_tokens"][1:-1]

            if rel_tokens:
                starting_indexes = [j for j in range(len(extraction["tokens"])) if starts_with(
                    rel_tokens, extraction["tokens"], j)]
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

                        assert rel_tokens == extraction['tokens'][final_idx: final_idx + len(
                            rel_tokens)]
                        extraction['rel_tagged'] = True
                        extraction['tags'][final_idx: final_idx +
                                        len(rel_tokens)] = ['REL'] * len(rel_tokens)

                    else:
                        arg2_idx = extraction['tags'].index('ARG2')
                        final_idx = -1
                        for idx in starting_indexes:
                            dist = abs(arg1_idx - idx) + abs(arg2_idx - idx)
                            if dist < min_dist:
                                min_dist = dist
                                final_idx = idx

                        assert rel_tokens == extraction['tokens'][final_idx: final_idx + len(
                            rel_tokens)]
                        extraction['rel_tagged'] = True
                        extraction['tags'][final_idx: final_idx +
                                        len(rel_tokens)] = ['REL'] * len(rel_tokens)


def label_location(extractions):

    for extraction in extractions:
        if len(extraction["loc_args"]) == 1:
            matches = [difflib.SequenceMatcher(None, arg.strip().split(
            ), extraction["tokens"]).get_matching_blocks() for arg in extraction["loc_args"]]
            if all(len(match) == 2 for match in matches):
                if all(match[0].a == 0 and match[0].size == match[1].a and match[1].size == 0 for match in matches):
                    for match in matches:
                        extraction["tags"][match[0].b: match[0].b +
                                        match[0].size] = ["LOC"] * match[0].size


def label_time(extractions):

    for extraction in extractions:
        if len(extraction["time_args"]) == 1:
            matches = [difflib.SequenceMatcher(None, arg.strip().split(), extraction["tokens"]).get_matching_blocks() for arg in extraction["time_args"]]
            if all(len(match) == 2 for match in matches):
                if all (match[0].a == 0 and match[0].size == match[1].a and match[1].size == 0 for match in matches):
                    for match in matches:
                        extraction["tags"][match[0].b : match[0].b + match[0].size] = ["TIME"] * match[0].size


def get_num_extractions(extractions):

    count = 0
    for extraction in extractions:
        if extraction['arg2_tagged'] and extraction['rel_tagged'] and extraction['arg1_tagged']:
            if 'REL' in extraction['tags'] and 'ARG1' in extraction['tags']:
                if extraction['arg2'] == '' or 'ARG2' in extraction['tags']:
                    count += 1
    return count


def get_extraction(sentence, result_tuple):

    confidence, arg1, arg2, args, time_args, loc_args, rel = result_tuple

    extraction = {}
    extraction["text"] = sentence.strip() + " [unused1] [unused2] [unused3]"
    extraction["tokens"] = extraction["text"].split()
    extraction["tags"] = ['NONE'] * len(extraction["tokens"])
    extraction["arg1"] = arg1.strip()
    extraction["arg1_tokens"] = extraction["arg1"].split()
    extraction["arg1_tagged"] = False
    extraction["rel"] = rel.strip()
    extraction["rel_tokens"] = extraction["rel"].split()
    extraction["rel_tagged"] = False
    extraction["arg2"] = arg2.strip()
    extraction["arg2_tokens"] = extraction["arg2"].split()
    if extraction["arg2"] == '':
        extraction["arg2_tokens"] = []
    extraction["arg2_tagged"] = False
    extraction["args"] = args
    extraction["time_args"] = time_args
    extraction["loc_args"] = loc_args
    extraction["args_tokens"] = []
    for arg in extraction["args"]:
        extraction["args_tokens"] = extraction["args_tokens"] + \
            arg.strip().split()
    if len(extraction["args"]) == 1 and extraction["args"][0] == '':
        extraction["args"] = []
        extraction["args_tokens"] = []
    extraction["time_args_tokens"] = []
    for arg in extraction["time_args"]:
        extraction["time_args_tokens"] = extraction["time_args_tokens"] + \
            arg.strip().split()
    extraction["loc_args_tokens"] = []
    for arg in extraction["loc_args"]:
        extraction["loc_args_tokens"] = extraction["loc_args_tokens"] + \
            arg.strip().split()
    extraction["confidence"] = confidence

    return extraction


def parse_result_text(text):
    if re.match("\d\.\d\d\s\(.*\)", text):
        confidence = float(text.split()[0])

        # text = re.sub("\A\d.\d\d\s", "", text).strip("()").strip().split(";")
        text = re.sub("\A\d.\d\d\s", "", text)[1:-1].split(";")

    elif re.match("\d\.\d\d\sContext\(.*\)\:", text):
        confidence = float(text.split()[0])

        text = re.sub("\A\d.\d\d\sContext\(.*\)\:", "", text)[1:-1].split(";")
    else:
        ipdb.set_trace()
        assert False

    if len(text) >= 3:
        arg1 = text[0].strip()
        rel = text[1].strip()
        arg2 = ''
        args = []
        time_args = []
        loc_args = []
        if text[2].strip().startswith('T:'):
            arg2 = text[2].strip()[2:]
        elif text[2].strip().startswith('L:'):
            arg2 = text[2].strip()[2:]
        else:
            arg2 = text[2].strip()
        for token in text[3:]:
            if token.strip().startswith('T:'):
                time_args.append(token.strip()[2:])
            elif token.strip().startswith('L:'):
                loc_args.append(token.strip()[2:])
            else:
                args.append(token.strip())

        return (confidence, arg1, arg2, args, time_args, loc_args, rel)
    else:
        assert False


def process_merge_openie(lines, sentences, conj_mapping, selected_extractions):
    wiki_extractions = []
    wiki_sentences = set()
    extraction_num = -1

    for line in lines:
        if len(line) > 0 and len(line.split('\n')) > 0:
            extraction_text = line.split('\n')
            sentence = extraction_text[0]
            # if (sentence in sentences or sentence in conj_mapping) and sentence not in wiki_sentences:
            if sentence in sentences or sentence in conj_mapping:    
                if conj_mapping != None and sentence in conj_mapping.values(): # ignore extractions of original sentence - when splits are available
                    continue
                wiki_sentences.add(sentence)
                if conj_mapping != None and sentence in conj_mapping:
                    sentence = conj_mapping[sentence]

                for i in range(1, len(extraction_text)):
                    extraction_num += 1
                    # print(extraction_num, extraction_text[i])
                    if extraction_num in selected_extractions or conj_mapping == None:
                        extraction = get_extraction(sentence, parse_result_text(extraction_text[i]))
                        wiki_extractions.append(extraction)
                    
                    # arg1_rel_tokens = f'{" ".join(extraction["arg1_tokens"])} {" ".join(extraction["rel_tokens"])}'
                    # ipdb.set_trace()
                    # if arg1_rel_tokens in extractionsD[sentence]:

    return wiki_extractions

def load_conj_mapping(conj_fp):
    conj_mapping = dict()
    conj_mapping_values = set()
    content = open(conj_fp).read()
    for example in content.split('\n\n'):
        for i, line in enumerate(example.strip('\n').split('\n')):
            if i == 0:
                orig_sentence = line
            else:
                conj_mapping[line] = orig_sentence
    conj_mapping_values = conj_mapping.values()
    return conj_mapping

def process_allennlp(lines):
    for i in range(0, len(lines)):
        lines[i] = lines[i].strip().split('\t')
        lines[i][2] = float(lines[i][2])

    extractions, sentences, selected_extractions = [], set(), set()
    for line in lines:
        extraction = {}
        # assert len(line) == 3
        assert len(re.findall("<arg1>.*</arg1>", line[1])) == 1
        assert len(re.findall("<rel>.*</rel>", line[1])) == 1
        assert len(re.findall("<arg2>.*</arg2>", line[1])) == 1

        extraction["text"] = line[0]
        extraction["tokens"] = line[0].strip().split()
        extraction["tags"] = ['NONE']*len(extraction["tokens"])
        for arg in ['arg1', 'rel', 'arg2']:
            begin_tag, end_tag = '<' + arg + '>', '</' + arg + '>'
            extraction[arg] = ' '.join(re.findall(
                begin_tag + '.*' + end_tag, line[1])[0].strip(begin_tag).strip(end_tag).strip().split())
            extraction[arg + '_tokens'] = extraction[arg].strip().split()
            extraction[arg + '_tagged'] = False
        
        extraction["confidence"] = line[2]
        extractions.append(extraction)
        sentences.add(extraction["text"])

        if len(line) == 4:
            selected_extractions.add(int(line[3]))
        # if extraction["text"] not in extractionsD:
        #     extractionsD[extraction["text"]] = set()
        # extractionsD[extraction["text"]].add(f'{" ".join(extraction["arg1_tokens"])} {" ".join(extraction["rel_tokens"])}')

    return extractions, sentences, selected_extractions


def main(input_fp, output_fp, wiki_fp, conj_fp):
    
    print("Processing file " + input_fp + " ...")
    with open(input_fp, 'r') as f:
        lines = f.readlines()
    _, oie4_sentences, selected_extractions = process_allennlp(lines)
    print("Total number of extractions are " + str(len(lines)))

    conj_mapping = None
    if conj_fp != None:
        print("Loading conjunction mapping from " + conj_fp + " ...")
        conj_mapping = load_conj_mapping(conj_fp)

    print("Processing file " + wiki_fp + " ...")
    with open(wiki_fp, 'r') as f:
        lines = f.read().split("\n\n")
    wiki_extractions = process_merge_openie(lines, oie4_sentences, conj_mapping, selected_extractions)

    print("Labeling arg1, rel and arg2 ...")
    for extraction in wiki_extractions:
        label_arg2(extraction)
        label_arg(extraction, 'rel')
        label_arg(extraction, 'arg1')
    print("Number of extractions processed is " +
          str(get_num_extractions(wiki_extractions)))
    
    print("Labeling [is], [of], [from] ...")
    label_is_of_relations(wiki_extractions)
    print("Number of extractions processed is " +
          str(get_num_extractions(wiki_extractions)))

    print("Labeling extractions with multiple candidates of arg1 ...")
    label_multiple_arg1(wiki_extractions)
    print("Number of extractions processed is " +
          str(get_num_extractions(wiki_extractions)))

    print("Labeling extractions with multiple candidates of rel ...")
    label_multiple_rel(wiki_extractions)
    print("Number of extractions processed is " +
          str(get_num_extractions(wiki_extractions)))
    
    print("Labeling location ...")
    label_location(wiki_extractions)
    print("Number of extractions processed is " +
          str(get_num_extractions(wiki_extractions)))
    
    print("Labeling time ...")
    label_time(wiki_extractions)
    print("Number of extractions processed is " +
          str(get_num_extractions(wiki_extractions)))

    print("Writing in file " + output_fp + " ...")
    with open(output_fp, 'w') as f:
        current_tokens = []
        for extraction in wiki_extractions:
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

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", required=False,
                        default='oie4_extractions.tsv')
    parser.add_argument("--output_fp", required=False,
                        default='openie5_seq5_new')
    parser.add_argument("--wiki_fp", required=False,
                        default='wiki.txt.openie4.processed')
    parser.add_argument("--conj_fp", required=False)

    args = parser.parse_args()
    main(args.input_fp, args.output_fp, args.wiki_fp, args.conj_fp)
