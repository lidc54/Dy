# the file has 1000 classes range(1000)
# there are two avaliable items in every line

al=[]
bl=[]
image = '/home1/ImageNet_ILSVRC2012/train_label.txt'
with open(image) as f:
    aa=f.readline()
    while True:
        bb=aa.split()
        al.append(len(bb))
        bl.append(int(bb[1]))
        aa=f.readline()
        if not aa:break
print 'all item in every line:',set(al)
print 'avaliable class range(1000)-set(class):',set(range(1000))-set(bl)
print 'avaliable class -set(class)range(1000):',set(bl)-set(range(1000))