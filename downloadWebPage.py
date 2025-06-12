from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup as soup
#from XSNLPReader import IfpriConflictTrainReader
from XSNLPReader.readerPostProcessor import BertPostProcessor as DataPostProcessor
#from IFPRIMLLIB import IFPRIManager
from configobj import ConfigObj
import argparse
import sys
import os
from pathlib import Path
import math
import copy
import random
from collections import defaultdict
import requests
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.firefox.service import Service



#from word2number import w2n


class WebDownload:
    def __init__(self, config=None, **kwargs):
        self.options = Options()
        #self.options.headless = True
        #self.options.binary = "/usr/bin/firefox"
        #cap = DesiredCapabilities().FIREFOX
        #cap["marionette"] = False
        #service = Service('/home/gate/repos/MTID_MEDIA_ANALYSIS_CROP_DISEASE/geckodrive/geckodriver')
        self.options.add_argument("--headless")

        self.options.log.level = "trace"

        self.driver = webdriver.Firefox(options=self.options)
        self.readerPostProcessor = DataPostProcessor(['text'], 'label', config={})

    def close(self):
        self.driver.close()


    def downloadUrlFromTxt(self, text_file, outputPath):
        with open(text_file, 'r') as fin:
            all_urls = fin.readlines()
        i=1
        for each_url in all_urls:
            title, html, raw_html = self.downLoadWebPage(each_url.strip())
            print(html)
            text_output_file_name = os.path.join(outputPath, 'text', str(i)+'.txt')
            with open(text_output_file_name, 'w') as fo:
                fo.write(each_url+'\n')
                fo.write(str(title)+'\n')
                fo.write(html)
            
            html_output_file_name = os.path.join(outputPath, 'html', str(i)+'.html')
            with open(html_output_file_name, 'w') as fo:
                fo.write(each_url+'\n')
                fo.write(raw_html)
            i+=1


    def downLoadWebPage(self, url, sleep_max=2):
        html = ''
        raw_html = ''
        title = ''
        cleaned_txt = ''
        #try:
        if True:
            print('init:',url)
            self.driver.get(url)

            #response = requests.get(url)
            #final_url = response.url
            final_url = self.driver.current_url

            print('final:',final_url)
            self.driver.get(final_url)
            html = self.driver.page_source
            #print(html)
            raw_html = html
            page = soup(html, 'html.parser')
            ps = page.find_all('p')
            title = page.find_all('title')
            if len(title) > 0:
                title = title[0]
            else:
                title = 'no title'
            cleaned_title = self.readerPostProcessor.scholarTextClean(str(title), strip_html=True, lower=False)
            all_p_tag_sentence = [ptag.text for ptag in ps]
            html = '\n'.join(all_p_tag_sentence)
            #print(html)
            cleaned_txt = self.readerPostProcessor.scholarTextClean(html, strip_html=True, lower=False)
            #print(cleaned_txt)
        #except:
        #    print('error prosess ', url)

        return title, html, raw_html, cleaned_txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_text", help="path to html link txt file")
    parser.add_argument("--url", help="path to html link txt file")
    parser.add_argument("--output", help="path to outputs")
    args = parser.parse_args()

    webManager = WebDownload()
    title, html, raw_html, clean_text = webManager.downLoadWebPage(args.url)
    #print(html)
    print(title)
    print(clean_text)

    webManager.close()


    if args.output:
        html_output_path = os.path.join(args.output,'html')
        text_output_path = os.path.join(args.output,'text')

        if not os.path.exists(html_output_path):
            outputPath = Path(html_output_path)
            outputPath.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(text_output_path):
            outputPath = Path(text_output_path)
            outputPath.mkdir(parents=True, exist_ok=True)

    #webManager.downloadUrlFromTxt(args.url_text, args.output)








