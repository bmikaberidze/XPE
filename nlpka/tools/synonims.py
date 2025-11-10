#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Sep 23 13:14:08 2020
@author: guga
"""

# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
from requests_html import HTMLSession

class Synonims():

    session = HTMLSession() # to keep all cookies
    URL = 'http://www.nplg.gov.ge/gwdict/index.php?a=index&d=17'
    data =  {
        '__VIEWSTATE': '/wEPDwUKMTU0NTIyNzUzNGQYAQUeX19Db250cm9sc1JlcXVpcmVQb3N0QmFja0tleV9fFgEFBmNoZWNrMcovfjNpJUpdf1MVrP5JwT1L5pjsL567PXNHM8jDeCwh',
        '__VIEWSTATEGENERATOR': '9E5244AD',
        '__EVENTVALIDATION': '/wEdAAVtSc+7zzq43mXnuG4HVOdfESCFkFW/RuhzY1oLb/NUVM34O/GfAV4V4n0wgFZHr3c/WMskeKo19Gyidl+m11dTn8hhIJObUkJ9Sisl+XJ2QuIfoevh8rcqZrgXTtqT3kVx2OASorM07VoZhC+LDkTh',
        'Button1': 'Analyze'
    }

    def __init__(self, word):
        html = self._submit(word)
        
    def _submit(self, entry):
        ''' 
        submit the entry to the web and obtain the reslting html 
        '''
        self.data['TextBox1'] = entry
        if self.filter:
            self.data['check1'] = 'on'        
        res = self.session.post(self.URL, data=self.data)
        return res.content    