import nlpka.tools.common as common
common.info(__file__, __name__, __package__)

import re

import nltk
nltk.download('punkt')

from string import digits
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

import nlpka.datasets.storage.collections.abbreviations as abbr
abbr_path = common.get_module_location(abbr)

class KaSenTok:

    tokenizer = None

    abbrEnds = {}
    abbreviations = set()

    wikiAbbrsFilename = f'{abbr_path}/wiki.abbrs.txt'
    abbrEndsFilename = f'{abbr_path}/abbr.ends.txt'

    def __init__(self):

        self.custom_sent_ends_tt = str.maketrans({'\n': '. ', '\t': '. '})
        self.remove_digits_tt = str.maketrans('', '', digits)

        with open(self.wikiAbbrsFilename) as wikiAbbrsFile, \
             open(self.abbrEndsFilename) as abbrEndsFile:

            for a in wikiAbbrsFile:
                self.abbreviations.add(a[:-2])

            for ae in abbrEndsFile:
                self.abbrEnds[ae[:-1]] = 1

        punktParam = PunktParameters()
        punktParam.abbrev_types = self.abbreviations
        self.tokenizer = PunktSentenceTokenizer(punktParam)

    def tokenize(self, text):
        # Consider \n and \t as sentence endings
        text = text.translate(self.custom_sent_ends_tt)
        sentences = self.tokenizer.tokenize(text)
        removeIndxStack = []
        for i in range(1, len(sentences)):
            sentence = sentences[i-1]
            if sentence[-1] == '.':
                lastToken = sentence[sentence.rfind(' ')+1:]
                lastToken = lastToken[lastToken.rfind(' ')+1:]
                lastWords = word_tokenize(lastToken)
                if len(lastWords) > 1:
                    lastWord = lastWords[-2]
                    lastPart = lastWord[lastWord.rfind('.')+1:]
                    lastPart = lastPart.translate(self.remove_digits_tt)
                    if lastPart in self.abbrEnds:
                        removeIndxStack.append(i-1)
                        sentences[i] = sentences[i-1] + ' ' + sentences[i]
                        # print(lastPart, word_tokenize(lastToken))

        while removeIndxStack:
            i = removeIndxStack.pop()
            del sentences[i]

        return sentences

if (__name__ == '__main__'):
    kaSenTok = KaSenTok()
    text = 'სიმღერების სია, ე.ერნ. \
აშშ გამოცემა. \
მათ სანაცვლოდ წარმოდგენილია: \
1867წ. ავსტრია-უნგრეთის კომპრომისი (გერმ. \
Kiegyezés) — შეთანხმება\n მონარქია.'
 
    print(sent_tokenize(text))
    print(kaSenTok.tokenizer.tokenize(text))
    print(kaSenTok.tokenize(text))
