import os, operator
import random
import sys

random.seed(666)

labels={'positive':'1', 'negative':'0'}

def write_to_file(data, prefix):

    sindices=range(len(data[0]))
    random.shuffle(sindices)
    data=[[data[0][k] for k in sindices], [data[1][k] for k in sindices]]
    labfile=open(prefix + '.lab', 'w')
    for l in data[0]:
        labfile.write(l+'\n')
    labfile.close()
    vecfile=open(prefix + '.vec', 'w')
    for l2 in data[1]:
        vecfile.write(l2+'\n')
    vecfile.close()

def write_to_file_unlab(data, prefix):

    sindices=range(len(data))
    random.shuffle(sindices)
    data=[data[k] for k in sindices]
    vecfile=open(prefix + '.vec', 'w')
    for l2 in data:
        vecfile.write(l2+'\n')
    vecfile.close()


def convert_into_libsvm(this_ex_list, feats_list):
    this_ex_dict={}
    for feat in this_ex_list:
      this_ex_dict[feat.rpartition(':')[0]]=feat.rpartition(':')[2]
    res=''
    for afeat in feats_list:
        if afeat[0] in this_ex_dict:
            res += afeat[1] + ':' + this_ex_dict[afeat[0]] + ' '
    return res



def PreProcess(DictPath,FullAmazonFile,SmallAmazonFile,DictSize,FullAmazonPath,SmallAmazonPath):
    files=dict()
    listdomain = os.listdir(SmallAmazonPath)
    for i in listdomain:
        files[i]=[]
        files[i]+=[SmallAmazonPath+i+'/negative.review']
        files[i]+=[SmallAmazonPath+i+'/positive.review']
    
    # get features
    features={}
    for domain in files:
        for datafile in files[domain]:
            for line in open(datafile):
                ex=line.split()[:-1] # label is last
                for feat in ex:
                    feat_idx=feat.rpartition(':')[0]
                    if feat_idx not in features:
                        features[feat_idx]=0
                    features[feat_idx]+=int(feat.rpartition(':')[2])
    
    # sort then by frequency
    sortedFeatList = sorted(features.items(),key=operator.itemgetter(1),reverse=True)
    
    # filter to keep the most frequent
    accepted_feats=[]
    ffile=open(DictPath, 'w')
    for i in range(DictSize):
        accepted_feats+=[[sortedFeatList[i][0], str(i)]]
        ffile.write(str(i+1)+' '+sortedFeatList[i][0]+' '+str(sortedFeatList[i][1])+'\n')
    ffile.close()

    # create Small Amazon files
    all_trnex=[[],[]]
    for domain in files:
        for datafile in files[domain]:
            cnt=0
            for line in open(datafile):
                ex=convert_into_libsvm(line.split()[:-1], accepted_feats)
                lab=labels[line.rpartition(':')[2].strip()]
                all_trnex[0]+=[lab]
                all_trnex[1]+=[ex]
                cnt+=1

    write_to_file(all_trnex, SmallAmazonFile)

    # create Full Amazon files
    files=dict()
    listdomain = os.listdir(FullAmazonPath)
    for i in listdomain:
        if i not in ['stopwords','summary.txt']:
            files[i]=[]
            files[i]+=[FullAmazonPath+i+'/processed.review']
    
    all_trnex=[[],[]]
    for domain in files:
        for datafile in files[domain]:
            cnt=0
            for line in open(datafile):
                ex=convert_into_libsvm(line.split()[:-1], accepted_feats)
                lab=labels[line.rpartition(':')[2].strip()]
                all_trnex[0]+=[lab]
                all_trnex[1]+=[ex]
                cnt+=1
    write_to_file(all_trnex, FullAmazonFile)

if __name__ == '__main__':
    if len(sys.argv) <= 6:
        print "Usage:", sys.argv[0], "DictPath FullAmazonFile SmallAmazonFile DictSize FullAmazonPath SmallAmazonPath"
        sys.exit(-1)
    DictPath = sys.argv[1]
    FullAmazonFile = sys.argv[2]
    SmallAmazonFile = sys.argv[3]
    DictSize = int(sys.argv[4])
    FullAmazonPath = sys.argv[5]
    SmallAmazonPath = sys.argv[6]
    PreProcess(DictPath,FullAmazonFile,SmallAmazonFile,DictSize,FullAmazonPath,SmallAmazonPath)
