import os
from pyltp import Segmentor
from gensim import corpora, models
import gensim
import random
from sklearn.neural_network import MLPClassifier
LAYER_SIZE = (5, 2)


def genData():
    path = "/home/liberty/Sentiment/sentiment-data/pnn_annotated.txt"
    MODELDIR = "/home/liberty/ltp_data"
    segmentor = Segmentor()
    segmentor.load(os.path.join(MODELDIR, "cws.model"))
    posList = []
    senList = []
    with open(path, "r") as file:
        with open("/home/liberty/Sentiment/sentiment-data/After.txt",
                  "w") as out:
            with open("/home/liberty/Sentiment/sentiment-data/Pos.txt",
                      "w") as posOut:
                cnt = 0
                for line in file.readlines():
                    random.seed(cnt * 10)
                    pos, sentence = line.split("\t")
                    words = list(segmentor.segment(sentence))
                    if cnt < 2500:
                        length = len(words)
                        unks = int(length * 0.1)
                        for i in range(unks):
                            idx = random.randint(0, length - 1)
                            words[idx] = "UNK"
                    senList.append(words)
                    posList.append(eval(pos))
                    out.write(" ".join(words) + "\n")
                    posOut.write(pos + "\n")
                    cnt += 1
            segmentor.release()
    return posList, senList


def readData():
    path = "/home/liberty/Sentiment/sentiment-data/After.txt"
    posPath = "/home/liberty/Sentiment/sentiment-data/Pos.txt"
    posList = []
    senList = []
    with open(path, "r") as senIn:
        with open(posPath, "r") as posIn:
            for (sen, pos) in zip(senIn.readlines(), posIn.readlines()):
                pos = eval(pos.replace("\n", ""))
                if pos == -1:
                    posList.append([1, 0, 0])
                elif pos == 0:
                    posList.append([0, 1, 0])
                else:
                    posList.append([0, 0, 1])
                senList.append(sen.lower().replace("\n", "").split(" "))
    return posList, senList


posList, senList = readData()
TrainList = senList[:2500]
TrainPosList = posList[:2500]
TestList = senList[2500:]
TestPosList = posList[2500:]
dictionary = corpora.Dictionary(TrainList)


def genTestList(dictionary, TestList):
    ans = []
    Dict = dictionary.token2id.keys()
    for sen in TestList:
        new_sen = [word if word in Dict else "UNK" for word in sen]
        ans.append(new_sen)
    return ans


def cmp(l1, l2):
    for i1, i2 in zip(l1, l2):
        if i1 != i2:
            return False
    return True


def judge(ans, TestPosList):
    cnt = 0
    correct = 0
    for ans_i, test_i in zip(ans, TestPosList):
        cnt += 1
        if cmp(list(ans_i), test_i):
            correct += 1
    print(correct, cnt)
    return correct / cnt


TestList = genTestList(dictionary, TestList)
bow_corpus = [dictionary.doc2bow(text) for text in TrainList]
tf_idf = models.TfidfModel(bow_corpus)
TF_Corpus = tf_idf[bow_corpus]

matrix = gensim.matutils.corpus2dense(bow_corpus, len(dictionary)).T

clf_1 = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=LAYER_SIZE, random_state=1)
#    max_iter=1,
#    warm_start=True)
test_corpus = [dictionary.doc2bow(txt) for txt in TestList]
test_matrix = gensim.matutils.corpus2dense(test_corpus, len(dictionary)).T
print(test_matrix.shape)
print("BOW:")
'''
for i in range(100):
    print("epoch:" + str(i))
    clf_1.fit(matrix, TrainPosList)
    test_ans = clf_1.predict(test_matrix)
    print(judge(test_ans, TestPosList))
'''
clf_1.fit(matrix, TrainPosList)
test_ans = clf_1.predict(test_matrix)
print(judge(test_ans, TestPosList))
matrix = gensim.matutils.corpus2dense(TF_Corpus, len(dictionary)).T

clf_2 = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=LAYER_SIZE, random_state=1)
#    max_iter=1,
#   warm_start=True)

test_tf_idf = models.TfidfModel(test_corpus)
test_TFIDF = test_tf_idf[test_corpus]

test_matrix = gensim.matutils.corpus2dense(test_TFIDF, len(dictionary)).T
print("TF-IDF:")
'''
for i in range(300):
    print("epoch:" + str(i))
    clf_2.fit(matrix, TrainPosList)
    test_ans = clf_2.predict(test_matrix)
    print(judge(test_ans, TestPosList))
'''
clf_2.fit(matrix, TrainPosList)
test_ans = clf_2.predict(test_matrix)
print(judge(test_ans, TestPosList))
