import csv

lines = csv.reader(open('wheather.csv','r'))
dataset = list(lines)

hypothesis = ['0']*6

for i in range(4):
    if(dataset[i][-1]=='Yes'):
        for j in range(6):
            if(hypothesis[j]=='0'):
                hypothesis[j] = dataset[i][j]
            elif(hypothesis[j]!=dataset[i][j]):
                hypothesis[j]='?'
            else:
                pass
    print(i+1, "=", hypothesis)