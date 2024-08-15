#!/usr/bin/python

import os
import sys
import re


def getFITS(fn):
    data = open(fn).read()
    starts = [s.start() for s in re.finditer("HREF",data)]
    urls = []
    for s in starts:
        actualStart = data.find("\"",s)+1
        actualEnd = data.find("\"",actualStart+1)
        url = data[actualStart:actualEnd]
        
        if any(url.endswith(post) for post in ["_err.fits","lat.fits","lon.fits"]):
            continue
        urls.append(url) 
    return urls

if __name__ == "__main__":
    # ex: python get_iquv.py ./Input_Data https://jsoc1.stanford.edu/SUM14/D1776011447/S00000/
    url = sys.argv[2]
    location = sys.argv[1]
    if not os.path.exists(location):
        os.makedirs(location)
    os.system("wget %s -O data.txt" % url)
    urls = getFITS("data.txt")
    for u in urls:
        print(u)
        os.system("cd %s && wget %s" % (location, u))
