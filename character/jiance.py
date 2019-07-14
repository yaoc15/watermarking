import os
result = []
for line in open("japan.txt"):
    line = line.strip('\n')
    list = line.split(" ")
    #print(line)
    if (float(list[1]) < 0.1):
        result.append(list)
        print(list[0][:-4])        
print(len(result))
print(result)

