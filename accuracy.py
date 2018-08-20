from __future__ import division
import json,sys
def precisionRecall(type,line,devline,index):
    true_positive=0
    false_negative=0
    false_positive=0
    for i in range(0,len(line)):
        if line[i][index] == devline[i][index] and devline[i][index] ==type:
            true_positive+=1
        elif devline[i][index]==type and line[i][index]!=devline[i][index]:
            false_negative+=1

        elif line[i][index]==type and line[i][index]!=devline[i][index]:
            false_positive+=1
    precision = true_positive/ float(true_positive +false_positive)
    recall = true_positive / float(true_positive +false_negative)
    f1 = 2*precision*recall /float( precision+recall)
    print precision , recall, f1, "for type", type
    return f1

def  main(argv):

    line=[]
    with open("percepoutput.txt",'r') as model:
        for eachline in model.readlines():
            eachline =eachline.decode(encoding ="utf-8")

            line.append(eachline.replace("\n", "").split(" "))
    print "printing", line[1]
    devline=[]
    with open("data/dev-key.txt",'r') as actual:
        for eachline in actual.readlines():
            eachline =eachline.decode(encoding ="utf-8")
            devline.append(eachline.replace("\n", "").split(" "))
    count =0
    totalcount=0
    f =open("error.txt", "w")
    true_positive = 0
    f1=[]
    f1.append(precisionRecall('Pos', line,devline,2))
    f1.append(precisionRecall('Neg',line,devline,2))
    f1.append(precisionRecall('True', line, devline, 1))
    f1.append(precisionRecall('Fake', line, devline, 1))
    average_f1 = sum(f1)/float(len(f1))
    print "f1",average_f1

    # for i in range(0,len(line)):
    #     for j in range(0,len(line[i])):
    #         totalcount+=1
    #         if line[i][j].strip() == devline[i][j].strip():
    #             count +=1
    #         else:
    #              f.write(line[i][j].encode("utf-8")+ " not matching with " + devline[i][j].encode("utf-8")+"\n")
    #
    #
    # print "accuracy is " , (count/totalcount)*100

if __name__ == "__main__":
    main(sys.argv)

# Results:
# Neg 0.49 0.78 0.60
# True 0.34 0.07 0.12
# Pos 0.47 0.20 0.28
# Fake 0.48 0.86 0.62
# Mean F1: 0.4054
# {'Neg': {'fp': 128, 'fn': 36, 'tp': 124}, 'True': {'fp': 23, 'fn': 148, 'tp': 12}, 'Pos': {'fp': 36, 'fn': 128, 'tp': 32}, 'Fake': {'fp': 148, 'fn': 23, 'tp': 137}}
