class Srudent(object):
    count = 0
    def __init__(self,name):
        Srudent.count +=1
        self.name = name

Srudent(1)
Srudent(1)
print(Srudent.count)
