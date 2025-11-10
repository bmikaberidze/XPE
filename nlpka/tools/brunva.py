#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Sep 23 13:14:08 2020
@author: guga
"""
import nlpka.tools.common as common
common.info(__file__, __name__, __package__)

from bs4 import BeautifulSoup
# from urllib.parse import urljoin
from requests_html import HTMLSession

class Brunva():

    session = HTMLSession() # to keep all cookies
    URL = 'http://glanguage.geoanbani.com/Corpus/Analyze/CorpusAnalyzer.aspx'
    data = {
        '__VIEWSTATE': '/wEPDwUKMTU0NTIyNzUzNGQYAQUeX19Db250cm9sc1JlcXVpcmVQb3N0QmFja0tleV9fFgEFBmNoZWNrMcovfjNpJUpdf1MVrP5JwT1L5pjsL567PXNHM8jDeCwh',
        '__VIEWSTATEGENERATOR': '9E5244AD',
        '__EVENTVALIDATION': '/wEdAAVtSc+7zzq43mXnuG4HVOdfESCFkFW/RuhzY1oLb/NUVM34O/GfAV4V4n0wgFZHr3c/WMskeKo19Gyidl+m11dTn8hhIJObUkJ9Sisl+XJ2QuIfoevh8rcqZrgXTtqT3kVx2OASorM07VoZhC+LDkTh',
        'Button1': 'Analyze'
    }

    # -------------------------------------------------------
    def __init__(self, filtr=True, scrape_frequencies=False):
        ''' 
        obtains different "brunebebi" of the given word. I dont know what 
        filter does, but it ticks the checkbox with the same name on the web.
        '''

        self.filter = filtr
        
        # I am keeping this optional to spare the memory resources
        self.scrape_frequencies = scrape_frequencies 
        
        # the number of entries found for this word (e.g. მზე returns 2).
        # for now I ignore everything but the first entry (not perfect)
        self.number_of_entries_found = 0
            
    # -------------------------------------------------------
    def brun(self, word, filter=-1, scrape_frequencies=-1):

        self.filter = self.filter if filter == -1 else filter  
        self.scrape_frequencies = self.scrape_frequencies if scrape_frequencies == -1 else scrape_frequencies  
        
        return self._parse_results(self._submit(word))

    # -------------------------------------------------------
    def _submit(self, entry):
        ''' 
        submit the entry to the web and obtain the reslting html 
        '''

        self.data['TextBox1'] = entry
            
        if self.filter:
            self.data['check1'] = 'on'
        
        res = self.session.post(self.URL, data=self.data)

        return res.content

    # -------------------------------------------------------
    def _parse_results(self, html):
        ''' 
        parse the first two rows of the table and and record the relevant 
        data into the object variables 
        '''
        
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find_all('table')[0]
        table_rows = table.find_all('tr')
        
        self.number_of_entries_found = len(table_rows)/2
        if self.number_of_entries_found < 1:
            return
        
        tr0 =  table_rows[0]
        tr1 =  table_rows[1]
        tds0 = tr0.find_all('td')
        row0 = [i.text for i in tds0]
        tds1 = tr1.find_all('td')
        row1 = [i.text for i in tds1]

        res = {}
        res['base'] = row0[1]
        res['forms'] = ['']*14
        if self.scrape_frequencies:
            res['form_frequencies'] = {}

        for i in range(14):
            form = row0[5+i]
            freq = row1[2+i]
            res['forms'][i] = form
            if self.scrape_frequencies:
                res['form_frequencies'][form] = int(freq)
        
        return res

### if it is executing as standalone script
# ანუ თუ სკრიბტად უშვებ, მაგალითად კონსოლიდან, მაშინ შევა იფში
# და თუ მოდულად აიმპორტებ, მაგალითად რომელიმე სხვა ფაილში, მაშინ არ შევა ამ იფში
if (__name__ == '__main__'):

    word = 'მზე'
    karuseli = Brunva(True, True)

    import sys
    if len(sys.argv) > 1:
        word = sys.argv[1]

    print('პასუხი:', karuseli.brun(word))