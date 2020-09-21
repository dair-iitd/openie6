import ipdb
import os
import csv

def get_extraction(sentence, labels):
    assert len(sentence) == len(labels)
    e1, r, e2 = '', '', ''
    for label_indx, label in enumerate(labels):
        word = sentence[label_indx]
        if label == 'E1':
            e1 += ' '+word
        if label == 'R':
            r += ' '+word
        if label == 'E2':
            e2 += ' '+word
    sentence = ' '.join(sentence).strip()
    e1, r, e2 = e1.strip(), r.strip(), e2.strip()
    extraction = '<arg1> '+e1+' </arg1> <rel> '+r+' </rel> <arg2> '+e2+' </arg2>'
    return sentence, extraction

def main():
    predD = dict()
    sense_carb = open('models/traditional/test/senseoie.tsv','w')
    test_sents = [s.replace(' ','') for s in open('carb/data/test.txt','r').readlines()]
    test_sents = open('carb/data/test.txt').readlines()
    testD = dict()
    for sent in test_sents:
        sent = sent.strip('\n').split('[unused1]')[0].strip()
        testD[sent.replace(' ','').lower()] = sent

    found_sentences = set()
    for row in csv.reader(open('models/traditional/SenseOIE_Output.csv','r')):
        if row[0] == 'sentence':
            sentence = row[1:]
        if row[0] == 'y_predicted':
            labels = row[1:]
            sentence, extraction = get_extraction(sentence, labels)
            norm_sentence = sentence.replace(' ','').lower()
            if norm_sentence in testD:
                sense_carb.write(testD[norm_sentence]+'\t'+extraction+'\t'+'1'+'\n')
                found_sentences.add(testD[norm_sentence])
                
    print('Found ',len(found_sentences),' extractions...')
    sense_carb.close()

    test_lines = open('carb/data/gold/test.tsv','r').readlines()
    filtered_test = open('sense_test.tsv','w')
    found_test = set()
    for line in test_lines:
        line = line.strip('\n')
        sentence = line.split('\t')[0]
        if sentence in found_sentences:
            filtered_test.write(line+'\n')
            found_test.add(sentence)
    filtered_test.close()
    print('Found ',len(found_test),' test sents')

main()
