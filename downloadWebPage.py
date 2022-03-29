from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup as soup
from XSNLPReader import IfpriConflictTrainReader
from XSNLPReader.readerPostProcessor import IfpriTrainPostProcessor
from IFPRIMLLIB import IFPRIManager
from configobj import ConfigObj
import argparse
import sys
import os
from pathlib import Path
import math
import copy
import random
from collections import defaultdict
from word2number import w2n


class WebDownload(IFPRIManager):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)


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
        readerPostProcessor = IfpriTrainPostProcessor(max_sent_len=150)
        options = Options()
        options.headless = True
        driver = webdriver.Firefox(options=options)
        html = ''
        raw_html = ''
        title = ''
        try:
            driver.get(url)
            html = driver.page_source
            raw_html = html
            page = soup(html, 'html.parser')
            ps = page.find_all('p')
            title = page.find_all('title')
            if len(title) > 0:
                title = title[0]
            else:
                title = 'no title'
            cleaned_title = readerPostProcessor.scholarTextClean(str(title), strip_html=True, lower=False)
            all_p_tag_sentence = [ptag.text for ptag in ps]
            html = '\n'.join(all_p_tag_sentence)
            cleaned_txt = readerPostProcessor.scholarTextClean(html, strip_html=True, lower=False)
        except:
            print('error prosess ', url)
        driver.close()

        return title, html, raw_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url_text", help="path to html link txt file")
    parser.add_argument("output", help="path to outputs")
    args = parser.parse_args()

    html_output_path = os.path.join(args.output,'html')
    text_output_path = os.path.join(args.output,'text')

    if not os.path.exists(html_output_path):
        outputPath = Path(html_output_path)
        outputPath.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(text_output_path):
        outputPath = Path(text_output_path)
        outputPath.mkdir(parents=True, exist_ok=True)

    webManager = WebDownload({})
    webManager.downloadUrlFromTxt(args.url_text, args.output)








