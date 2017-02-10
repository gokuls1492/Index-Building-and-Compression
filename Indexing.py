import glob
import re
import os
import string
import time
import math
import sys
import json
from array import array
from _collections import defaultdict
import operator
import pickle
from sys import getsizeof
from operator import itemgetter
from audioop import reverse
from string import replace
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import PorterStemmer

class DictEntry:
    def __init__(self,t,df,tf,pl):
        #self.entry = {'term':t,'docFreq':df,'termFreq':tf,'postingList':pl}
        self.term = t
        self.docFreq = df
        self.totTermFreq = tf
        self.postingList = pl
    #def __repr__(self):
    #    return "{}, {}, {}, {}".format(self.term, self.docFreq, self.totTermFreq, self.postingList)

class PostingEntry:
    def __init__(self,did,freq,maxtf,docl):
        #self.entry = {'term':t,'docFreq':df,'termFreq':tf,'postingList':pl}
        self.docId = did
        self.termFreq = freq
        self.maxTermFreq = maxtf
        self.docLen = docl 
    def __iter__(self):
        return self.__dict__.iteritems()


def getUnaryValue(leng):
    unaryValue = ""
    for i in range(0,leng):
        unaryValue += str(1)
    return unaryValue + str(0)

def getDeltaCode(num):
    binaryRep = str(bin(num))[2:]
    gammaCode = getGammaCode(len(binaryRep))
    offset = binaryRep[1:]
    deltaCode = gammaCode + offset
    return deltaCode

def getGammaCode(num):
    binaryRep = str(bin(num))[2:]
    offset = binaryRep[1:]
    unaryValue = getUnaryValue(len(offset))
    gammaCode = unaryValue + offset
    #byteGamma = bytearray(gammaCode)
    return gammaCode#byteGamma
    
def getMaxTermFreqDocLen(dictf):
    docLeng = 0
    maxTermFreq = 0
    for items in dictf:
        termFreq = dictf[items]
        docLeng += termFreq
        if not items in stopWords:
            if termFreq > maxTermFreq:
                maxTermFreq = termFreq
    return maxTermFreq,docLeng 
                
def insertDict(docId, token, termFreq, dictionary, maxTermFreq, docLeng):
    entry = dictionary.get(token)
    if entry is None:
        postingList = []
        entry = DictEntry(token,0,0,postingList)
        dictionary[token] = entry       

    entry.docFreq +=1
    postEntry = PostingEntry(docId,termFreq,maxTermFreq,docLeng)
    entry.postingList.append(postEntry)
    #entry.postingList[docId] = termFreq
    entry.totTermFreq+= termFreq
    #print termFreq
    
def Tokenize(files):
    textFile = open(files,"r")
    lText = textFile.read().lower()
    plainWord = re.sub('<[^>]*>','', lText)
    text = re.sub(r'\.(?![a-zA-Z]{3})', '', plainWord)
    text = text.replace("\'s","")
    #text = text.replace("\'[a-z]+",' ')
    words =re.sub('[^a-zA-Z]+', ' ', text).split()# re.split(r'[-=\.,?!:$;_()\[\]\`\'*"/\t\n\r\d+ \x0b\x0c]+', text)##re.sub(r"\p{P}+", "", text.lower()).split()#
    textFile.close()
    return [word.strip() for word in words if word.strip() != '']
        
    
def Lemmatiztion():        
    lmtzr = WordNetLemmatizer()
    for docId ,dfile in enumerate(filesList,1): 
        dictWord = {}        
        listWord = Tokenize(dfile)
        for word in listWord:
                tWord = lmtzr.lemmatize(word)
                #print tWord
                dictWord[tWord] = dictWord.get(tWord,0)+ 1
        
        mxtf, dolen = getMaxTermFreqDocLen(dictWord)
        for items in dictWord:
            termFreq = dictWord[items]
            if not items in stopWords :    
                insertDict(docId,items,termFreq,dictionary_uncomp_v1,mxtf,dolen)

        del dictWord
        
def Stemming():
    stmr = PorterStemmer()
    for docId ,dfile in enumerate(filesList,1):
        listWord = Tokenize(dfile)    
        stemWord = {}        
        for w in listWord:
            tempStem = stmr.stem(w) 
            stemWord[tempStem]= stemWord.get(tempStem,0)+1
        
        mxtf, dolen = getMaxTermFreqDocLen(stemWord)
        
        for items in stemWord:
            termFreq = stemWord[items]    
            if not items in stopWords :    
                insertDict(docId,items,termFreq,dictionary_uncomp_v2,mxtf,dolen)
        
        del stemWord
    #json.dump(stemWord, open("E:\output1.txt","w") )

def getDocLargestMaxTF(dictionary_uncomp):
    maz=0
    for term in dictionary_uncomp.keys():
        entry = dictionary_uncomp.get(term)
        for pEntry in entry.postingList:
            if pEntry.maxTermFreq > maz:
                maz = pEntry.maxTermFreq

    for term in dictionary_uncomp.keys():
        entry = dictionary_uncomp.get(term)
        for pEntry in entry.postingList:
            if pEntry.maxTermFreq == maz:            
                return pEntry.docId,pEntry.maxTermFreq
            
def getDocLargestDocLen(dictionary_uncomp):
    doc=0
    for term in dictionary_uncomp.keys():
        entry = dictionary_uncomp.get(term)
        for pEntry in entry.postingList:
            if pEntry.docLen > doc:
                doc = pEntry.docLen

    for term in dictionary_uncomp.keys():
        entry = dictionary_uncomp.get(term)
        for pEntry in entry.postingList:
            if pEntry.docLen == doc:            
                return pEntry.docId,pEntry.docLen
            

def blockedCompression():
    k=8
    tempK=0
    dictString = ""
    termFreqBlock = {}
    docFreqBlock = {}
    gammaEncodingList = []
    tempIndexList = []
    for f,term in enumerate(dictionary_uncomp_v1.keys()):
        if tempK < k:
            dictString += str(len(term)) + term
            entry = dictionary_uncomp_v1.get(term)
            prevId = 0
            pEntry = PostingEntry(0,0,0,0)
            for pEntry in entry.postingList:
                docId = getGammaCode(pEntry.docId - prevId)
                gammaEncodingList.append(docId)
                prevId = pEntry.docId
            termFreqBlock[tempK] = getGammaCode(entry.totTermFreq)
            docFreqBlock[tempK] = getGammaCode(entry.docFreq)
            tempK += 1
             
        if tempK == k or f == len(dictionary_uncomp_v1)-1:
            tempK=0
            tempIndexList.append(gammaEncodingList)
            tempIndexList.append(termFreqBlock)
            tempIndexList.append(docFreqBlock)
            compressedIndexV1[dictString] = tempIndexList 
            dictString = ""
            tempIndexList = []
            gammaEncodingList = []
            termFreqBlock = {}
            docFreqBlock = {}
                                
    return compressedIndexV1

def commonPrefix(m):
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1

def frontCoding():
    k=8
    tempK=0
    termFreqBlock = {}
    deltaEncodingList = []
    docFreqBlock = {}
    termList= []
    tempIndexList = []
    prefix = ""
    temp = ""
    
    for f,term in enumerate(sorted(dictionary_uncomp_v2.keys())):
        if tempK < k:
            termList.append(term)
            tempK += 1
        
        if tempK == k or f == len(dictionary_uncomp_v2)-1 :    
            prefix = commonPrefix(termList)
            if prefix:
                temp += "["
                for n,item in enumerate(termList):
                    if item.startswith(prefix):
                        if n == 0:
                            temp += str(len(item)) + prefix + "*" + item[len(prefix):]
                        if n > 0:
                            temp += str(len(item[len(prefix):])) + "|" + item[len(prefix):]                 
                    else:
                        if n == 0:
                            temp += str(len(item)) + prefix + "*" +  item[:]
                        if n > 0:
                            temp += str(len(item[:])) + "|" + item[:]
                    
                    entry = dictionary_uncomp_v2.get(item)
                    prevId = 0
                    pEntry = PostingEntry(0,0,0,0)
                    
                    for pEntry in entry.postingList:
                        docId = getGammaCode(pEntry.docId - prevId)
                        deltaEncodingList.append(docId)
                        prevId = pEntry.docId
                
                    termFreqBlock[n] = getDeltaCode(entry.totTermFreq)
                    docFreqBlock[n] = getDeltaCode(entry.docFreq)
               
                temp += "]"
                tempIndexList.append(deltaEncodingList) 
                tempIndexList.append(termFreqBlock)
                tempIndexList.append(docFreqBlock)
                compressedIndexV2[temp] = tempIndexList 
                tempK=0
                temp = ""
                tempIndexList = []
                termList = []
                deltaEncodingList = []
                termFreqBlock = {}
                docFreqBlock = {}
    return compressedIndexV2
                 
dirPath  = sys.argv[1]+"/*"#"D:\UTD\IR\HW1/Cranfield/*"#"D:\UTD\IR\HW1\New folder/*"#"/people/cs/s/sanda/cs6322/Cranfield/*"##
fileStop = open(sys.argv[2])
filesList = glob.glob(dirPath)
stopWords = fileStop.read()
compressedIndexV1 = defaultdict()
compressedIndexV2 = defaultdict()
dictionary_uncomp_v1 = dict()
dictionary_uncomp_v2 = dict()
dictWord = defaultdict(int)
stemWord = defaultdict(int)
docList = defaultdict(int)
start_time = time.time()
Lemmatiztion()
end_time = time.time()
print "Elapsed time to build Index version 1: "+str(end_time - start_time)+" seconds" 
start = time.time()
Stemming()
end = time.time()
print "Elapsed time to build Index version 2: "+str( end - start)+" seconds" 

with open('Index_Version1.uncompress', 'wb') as outfile:
    pickle.dump(dictionary_uncomp_v1, outfile, pickle.HIGHEST_PROTOCOL)

#print "Size of the Index version 1 uncompressed (in bytes): " + str(getSizeOfUncompIndex(dictionary_uncomp_v1))
print "Size of the Index version 1 uncompressed (in bytes): " + str(os.path.getsize("Index_Version1.uncompress"))

with open('Index_Version2.uncompress', 'wb') as outfile1:
    pickle.dump(dictionary_uncomp_v2, outfile1, pickle.HIGHEST_PROTOCOL)

print "Size of the Index version 2 uncompressed (in bytes): " + str(os.path.getsize("Index_Version2.uncompress"))
print ""
blockedCompression()
with open('Index_Version1.compress', 'wb') as outfile2:
    pickle.dump(compressedIndexV1, outfile2, pickle.HIGHEST_PROTOCOL)

frontCoding()
with open('Index_Version2.compress', 'wb') as outfile3:
    pickle.dump(compressedIndexV2, outfile3, pickle.HIGHEST_PROTOCOL)

print "Size of the Index version 1 compressed (in bytes): " + str(os.path.getsize("Index_Version1.compress"))
print "Size of the Index version 2 compressed (in bytes): " + str(os.path.getsize("Index_Version2.compress"))

#print "Size of the Index version 2 uncompressed (in bytes): " + str(getSizeOfUncompIndex(dictionary_uncomp_v2))
print ""
print "Number of inverted lists in version 1: " + str(len(dictionary_uncomp_v1))
print "Number of inverted lists in version 2: " + str(len(dictionary_uncomp_v2))

words = ["reynolds", "nasa", "prandtl", "flow", "pressure", "boundary", "shock"]
print ""
print '{0:7}  {1:7}  {2:7}  {3:7}'.format("Term ","Document-Frequency ","Total-Term-Freq ","Index-Size(bytes)")
#print "Term " +","+ "Document_Freq " +","+ "Total_Term_Freq "+","+"Inverted_Index_Size_bytes "
for i in words:
    entry = dictionary_uncomp_v1.get(i)
    print '{0:15}  {1:15}  {2:15}  {3:37}'.format(i,str(entry.docFreq),str(entry.totTermFreq),str(sys.getsizeof(entry.postingList)))#str(getSizeOfUncompPosting(entry.postingList)))
 
print ""
print '{0:10}  {1:15}'.format("Term-NASA","Document-Frequency")
for term in dictionary_uncomp_v1.keys():
    if term == "nasa":
        entry = dictionary_uncomp_v1.get(term)
        df = entry.docFreq
        #ttf = entry.totTermFreq
        print '{0:10}  {1:15}'.format(term,str(df))
        print "Posting Lists of NASA:" 
        print '{0:10}  {1:15}  {2:15}  {3:15}'.format("Doc-ID","Term-Freq", "Max-Term","Doc Len")
        for dlist in entry.postingList[:3]:
            n_docid = dlist.docId
            n_tf = dlist.termFreq
            n_max_tf = dlist.maxTermFreq
            n_len = dlist.docLen
            print '{0:10}  {1:15}  {2:15}  {3:15}'.format(str(n_docid),str(n_tf),str(n_max_tf),str(n_len))
                
maxDF =0

print ""
print "Term with the largest df from index 1 "
for term in dictionary_uncomp_v1.keys():
        entry = dictionary_uncomp_v1.get(term)
        if entry.docFreq > maxDF:
            maxDF = entry.docFreq
for term in dictionary_uncomp_v1.keys():
        entry = dictionary_uncomp_v1.get(term)
        if entry.docFreq == maxDF:
            print term +" - "+ str(maxDF)
maxDF =0
print ""
print "Term with the largest df from index 2 "
for term in dictionary_uncomp_v2.keys():
    entry = dictionary_uncomp_v2.get(term)
    if entry.docFreq > maxDF:
        maxDF = entry.docFreq
for term in dictionary_uncomp_v2.keys():
    entry = dictionary_uncomp_v2.get(term)        
    if entry.docFreq == maxDF:
        print term +" - "+ str(maxDF)

print ""
print "Document with largest max-tf :"
doc, mxterm = getDocLargestMaxTF(dictionary_uncomp_v1)
print "Cranfield0" +str(doc) + " - " + str(mxterm)
print ""
print "Document with largest doc-len :"
ndoc,ndoclen = getDocLargestDocLen(dictionary_uncomp_v1)
print "Cranfield0" +str(ndoc)+" - "+str(ndoclen)

minDF=maxDF
print ""
print "Terms with the lowest df from index 1 "
for term in dictionary_uncomp_v1.keys():
        entry = dictionary_uncomp_v1.get(term)
        if entry.docFreq < minDF:
            minDF = entry.docFreq
f = open('LowestDF_v1.txt', 'w')
print "Check file LowestDF_v1.txt for output"
for term in dictionary_uncomp_v1.keys():
        entry = dictionary_uncomp_v1.get(term)
        if entry.docFreq == minDF:
            #print term +" - "+ str(minDF)+",",
            f.write(term +" - "+ str(minDF)+",",)
f.close()
minDF=maxDF
print ""
print "Terms with the lowest df from index 2"
for term in dictionary_uncomp_v2.keys():
        entry = dictionary_uncomp_v2.get(term)
        if entry.docFreq < minDF:
            minDF = entry.docFreq
f = open('LowestDF_v2.txt', 'w')
print "Check file LowestDF_v2.txt for output"
for term in dictionary_uncomp_v2.keys():
        entry = dictionary_uncomp_v2.get(term)
        if entry.docFreq == minDF:
            #print term +" - "+ str(minDF)+",",
            f.write(term +" - "+ str(minDF)+",",)
f.close()
fileStop.close()