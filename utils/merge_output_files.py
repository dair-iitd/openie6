import re
import argparse

def contains(extr, list_extr):
    for i in range(0, len(list_extr)):
        if extr['arg1'] == list_extr[i]['arg1'] and extr['arg2'] == list_extr[i]['arg2'] and extr['rel'] == list_extr[i]['rel']:
            return True
    return False

def get_extraction(arg1, rel, arg2, confidence):
    return {'arg1': arg1, 'rel': rel, 'arg2': arg2, 'confidence': confidence}

def main(new_scores_file, original_scores_file, num_extractions):

    sentences = []

    with open(new_scores_file, 'r') as f:
        file = f.read()

    extractions = {}

    for l in file.split('\n'):
        if len(l.strip()) > 0:
            lines = l.strip().split('\t')
            assert len(lines) == 3
            assert len(re.findall("<arg1>.*</arg1>", lines[1])) == 1
            assert len(re.findall("<rel>.*</rel>", lines[1])) == 1
            assert len(re.findall("<arg2>.*</arg2>", lines[1])) == 1

            sentence = lines[0]
            if sentence not in sentences:
                sentences.append(sentence)

            arg1 = re.findall("<arg1>.*</arg1>", lines[1])[0].strip('<arg1>').strip('</arg1>').strip()
            rel = re.findall("<rel>.*</rel>", lines[1])[0].strip('<rel>').strip('</rel>').strip()
            arg2 = re.findall("<arg2>.*</arg2>", lines[1])[0].strip('<arg2>').strip('</arg2>').strip()
            confidence = float(lines[2])

            if sentence not in extractions.keys():
                extractions[sentence] = []
            extractions[sentence].append(get_extraction(arg1, rel, arg2, confidence))

    with open(original_scores_file, 'r') as f:
        file = f.read()
    
    for l in file.split('\n'):
        if len(l.strip()) > 0:
            lines = l.strip().split('\t')
            assert len(lines) == 3
            assert len(re.findall("<arg1>.*</arg1>", lines[1])) == 1
            assert len(re.findall("<rel>.*</rel>", lines[1])) == 1
            assert len(re.findall("<arg2>.*</arg2>", lines[1])) == 1

            sentence = lines[0]

            arg1 = re.findall("<arg1>.*</arg1>", lines[1])[0].strip('<arg1>').strip('</arg1>').strip()
            rel = re.findall("<rel>.*</rel>", lines[1])[0].strip('<rel>').strip('</rel>').strip()
            arg2 = re.findall("<arg2>.*</arg2>", lines[1])[0].strip('<arg2>').strip('</arg2>').strip()
            confidence = float(lines[2])

            extraction = get_extraction(arg1, rel, arg2, confidence)
            if sentence not in extractions:
                extractions[sentence] = []
            if sentence not in sentences:
                sentences.append(sentence)
            if not contains(extraction, extractions[sentence]):
                extractions[sentence].append(extraction)

    for sentence in extractions:
        extractions[sentence] = sorted(extractions[sentence], key=lambda x: x['confidence'], reverse=True)[:num_extractions]
    with open(new_scores_file + '.merged.'+str(num_extractions), 'w') as f:
        for sentence in sentences:
            for extraction in extractions[sentence]:
                f.write(sentence)
                f.write('\t')
                f.write('<arg1> ' + extraction['arg1'] + ' </arg1>')
                f.write(' ')
                f.write('<rel> ' + extraction['rel'] + ' </rel>')
                f.write(' ')
                f.write('<arg2> ' + extraction['arg2'] + ' </arg2>')
                f.write('\t')
                f.write(str(extraction['confidence']))
                f.write('\n')

    print("Output written to : " + new_scores_file + '.merged.'+str(num_extractions))
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_scores', help="file path for gold in allennlp format", required=True)
    parser.add_argument('--original', help="file path for gold in allennlp format", required=True)
    parser.add_argument('--num_extractions', help="number of final extractions", required=True, type=int)
    arguments = parser.parse_args()
    main(arguments.new_scores, arguments.original, arguments.num_extractions)