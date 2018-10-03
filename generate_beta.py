import cv2
import os
import random
import numpy as np
from PIL import Image
from operator import itemgetter
import time
from numpy.linalg import inv

start_time=time.clock()

refPt = []
pt=[]
cropping = False
n=1

def click_and_crop (event, x, y, flags, param ) :

    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        z=(x,y)
        #print 'initial '+str(z[0])+','+str(z[1])
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        z=(x,y)
        #print 'initial '+str(z[0])+','+str(z[1])
        cropping = False
        cv2.rectangle(image, refPt[0], refPt[1], (0, 0, 0), 2)
        cv2.imshow("image", image)


image = cv2.imread('a.png', 1)
img_size = image.shape
clone = image.copy()
recent = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
k=1
newpath = r'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img'
created=False
c=1
while created==False :
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print "New Folder Created"
        created=True
    else:
        c+=1
        newpath = r'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img'+str(c)

while True:

    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        image = recent.copy()

    elif key == ord("q"):
        break

    elif key == ord("c"):
        if len(refPt)==2:
            if n==1:
                pt=[refPt[0][0]]
                pt.append(refPt[0][1])
                pt.append(refPt[1][0])
                pt.append(refPt[1][1])

                #pt=[(refPt[0][0],refPt[0][1],refPt[1][0],refPt[1][1])]
            else:
                pt.append(refPt[0][0])
                pt.append(refPt[0][1])
                pt.append(refPt[1][0])
                pt.append(refPt[1][1])
                #pt.append((refPt[0][0],refPt[0][1],refPt[1][0],refPt[1][1]))
            n+=1
            cv2.rectangle(image, refPt[0],refPt[1], (0, 255, 0), 2)
            recent = image.copy()


    elif key==ord("x"):
        roi=clone[0:img_size[0],0:img_size[1]]
        path = r'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img'  + r'\accept'
        os.makedirs(path)
        path = r'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img'  + r'\reject'
        os.makedirs(path)
        sp1 = np.linspace(0, img_size[1] - 5, int((img_size[1] - 5) / 10))
        sp2 = np.linspace(0, img_size[0] - 5, int((img_size[0] - 5) / 10))
        count1=1
        count2=1
        for i in sp1:
            for j in sp2:
                #r1 = random.randrange(5, 18)
                #r2 = random.randrange(5, 18)
                r1=28
                r2=28
                k1 = int(j + r2)
                l1 = int(i + r1)
                i = int(i)
                j = int(j)
                if k1 <= img_size[0] and l1 <= img_size[1]:
                    crop = roi[j:k1, i:l1]
                elif k1 <= img_size[0]:
                    l1=img_size[1]
                    crop = roi[j:k1, i:img_size[1]]
                elif l1 <= img_size[1]:
                    k1=img_size[0]
                    crop = roi[j:img_size[0], i:l1]
                else:
                    k1=img_size[0]
                    l1=img_size[1]
                    crop = roi[j:img_size[0], i:img_size[1]]
                check=False
                for iq in range(0,len(pt),4):

                    x1=pt[iq]
                    y1=pt[iq+1]
                    x2=pt[iq+2]
                    y2=pt[iq+3]

                    if x1<=i and x2>=l1 and y1<=j and y2>=k1:
                        check=True
                        break
                if check==True:
                    y = "template" + str(count1)
                    count1 += 1
                    img=crop.shape
                    if img[0]==28 and img[1]==28:
                        cv2.imwrite('C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\accept' + '/' + y + '.png',
                                crop)
                    else:
                        count1-=1
                        continue
                    im = Image.open(
                        'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\accept' + '/' + y + '.png').convert(
                        'L')
                    pixels = im.getdata()  # get the pixels as a flattened sequence
                    black_thresh = 240
                    nblack = 0
                    for pixel in pixels:
                        if pixel < black_thresh:
                            nblack += 1
                    n = len(pixels)

                    if (nblack / float(n)) < 0.05:
                        p = r'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\accept' + '/' + y + '.png'
                        count1-=1
                        os.remove(p)

                else:
                    y = "template" + str(count2)
                    count2 += 1
                    img=crop.shape
                    if img[0]==28 and img[1]==28:
                        cv2.imwrite('C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\reject' + '/' + y + '.png',
                                crop)
                    else:
                        count2-=1
                        continue
                    im = Image.open(
                        'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\reject' + '/' + y + '.png').convert(
                        'L')
                    pixels = im.getdata()  # get the pixels as a flattened sequence
                    black_thresh = 240
                    nblack = 0
                    for pixel in pixels:
                        if pixel < black_thresh:
                            nblack += 1
                    n = len(pixels)

                    if (nblack / float(n)) < 0.05:
                        p = r'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\reject' + '/' + y + '.png'
                        count2-=1
                        os.remove(p)


cv2.destroyAllWindows()


#path = r'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img'  + r'\temp'
#os.makedirs(path)
ii=1
while ii<=500:
    r1 = random.randrange(0,10)
    r2 = random.randrange(0,10)
    i1 = random.randrange(1,50)
    x1 = random.randrange(5,17)
    y1 = random.randrange(5,17)
    y = "template" + str(i1)
    image = cv2.imread('C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\accept' + '/' + y + '.png', 1 )
    clone = image.copy()
    img_size = image.shape
    roi = clone[0:img_size[0], 0:img_size[1]]
    i = int(r1)
    j = int(r2)
    k1 = int(j + y1)
    l1 = int(i + x1)
    crop = roi[j:k1, i:l1]
    y = "template" + str(ii)
    cv2.imwrite('C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\temp' + '/' + y + '.png',
                crop)
    im = Image.open(
        'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\temp' + '/' + y + '.png').convert(
        'L')
    pixels = im.getdata()  # get the pixels as a flattened sequence
    black_thresh = 240
    nblack = 0
    for pixel in pixels:
        if pixel < black_thresh:
            nblack += 1
    n = len(pixels)

    if (nblack / float(n)) < 0.05:
        p = r'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\temp' + '/' + y + '.png'
        os.remove(p)
        ii-=1
        continue
    ii+=1

ii=501
while ii<=1000:
    r1 = random.randrange(0,10)
    r2 = random.randrange(0,10)
    i1 = random.randrange(1,900)
    x1 = random.randrange(5,17)
    y1 = random.randrange(5,17)
    y = "template" + str(i1)
    image = cv2.imread('C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\reject' + '/' + y + '.png', 1 )
    clone = image.copy()
    img_size = image.shape
    roi = clone[0:img_size[0], 0:img_size[1]]
    i = int(r1)
    j = int(r2)
    k1 = int(j + y1)
    l1 = int(i + x1)
    crop = roi[j:k1, i:l1]
    y = "template" + str(ii)
    cv2.imwrite('C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\temp' + '/' + y + '.png',
                crop)
    im = Image.open(
        'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\temp' + '/' + y + '.png').convert(
        'L')
    pixels = im.getdata()  # get the pixels as a flattened sequence
    black_thresh = 240
    nblack = 0
    for pixel in pixels:
        if pixel < black_thresh:
            nblack += 1
    n = len(pixels)

    if (nblack / float(n)) < 0.05:
        p = r'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\temp' + '/' + y + '.png'
        os.remove(p)
        ii-=1
        continue
    ii+=1


output_vector = [[0 for j in range(120)] for i in range(2)]


max_convulation = [[0 for j in range(120)] for i in range(1000)]
max = -90000


for j in range(1, 61):
    #y = "template" + str(j)
    output_vector[1][j-1]=1
    output_vector[0][j-1]=0
    '''template = cv2.imread(
        'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\accept' + '/' + y + '.png',
        0)
    for i in range(1, 1001):
        y = "template" + str(i)
        pool = cv2.imread('C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\temp' + '/' + y + '.png',
                          0)
        pool_size = pool.shape
        for m in range(1, 28-pool_size[0]):
            for n in range(1, 28 - pool_size[1]):
                sum = 0
                for p in range(1, pool_size[0]):
                    for q in range(1 , pool_size[1]):
                        sum += ((int(pool[p][q])-127)*(int(template[p+m][q+n])-127)*1.0)/(128*128)

                if sum > max:
                    max = sum

        max_convulation[i-1][j-1]=max
        max=-90000'''


max = -90000

for j in range(1, 61):
    #j1 = random.randrange(1, 1000)
    output_vector[1][j+59]=0
    output_vector[0][j+59]=1
    '''y = "template" + str(j1)
    template = cv2.imread('C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\reject' + '/' + y + '.png',
                          0)
    for i in range(1, 1001):
        y = "template" + str(i)
        pool = cv2.imread('C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\temp' + '/' + y + '.png',
                          0)
        pool_size = pool.shape
        for m in range(1, 28-pool_size[0]):
            for n in range(1, 28 - pool_size[1]):
                sum = 0
                for p in range(1, pool_size[0]):
                    for q in range(1 , pool_size[1]):
                        sum += ((int(pool[p][q])-127)*(int(template[p+m][q+n])-127)*1.0)/(128*128)

                if sum > max:
                    max = sum

        max_convulation[i-1][j+59]=max
        max=-90000'''

#print max_convulation
def function (i,j,mode):
    max = -90000

    #for j in range(1, 11):
    if mode==0:
        y = "template" + str(j+1)
        #output_vector[1][j - 1] = 1
        #output_vector[0][j - 1] = 0
        template = cv2.imread(
            'C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\accept' + '/' + y + '.png',
            0)
    else:
        j1 = random.randrange(1, 1000)
        #output_vector[1][j + 9] = 0
        #output_vector[0][j + 9] = 1
        y = "template" + str(j1)
        template = cv2.imread('C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\reject' + '/' + y + '.png',
                              0)
        #for i in range(1, 51):
    y = "template" + str(i)
    pool = cv2.imread('C:\Users\Aditya Agrawal\PycharmProjects\untitled\img' + r'\temp' + '/' + y + '.png',
                              0)
    pool_size = pool.shape

    for m in range(1, 28 - pool_size[0]):
        for n in range(1, 28 - pool_size[1]):
            sum = 0
            for p in range(1, pool_size[0]):
                for q in range(1, pool_size[1]):
                    sum += ((int(pool[p][q]) - 127) * (int(template[p + m][q + n]) - 127) * 1.0) / (128 * 128)

            if sum > max:
                max = sum
    max_convulation[i][j] = max
    max = -90000


T_selection=[]
count=0
for t_selection_size in range(1,11):
    T_pool = []
    a_k_i = [[0 for j in range(100)] for i in range(2)]
    influence=[]
    for t_pool_size in range(1,101):
        b= False
        while b==False:
            y=random.randrange(1,1000)
            if y in T_selection or y in T_pool:
                continue
            else:
                b=True
                T_pool.append(y)
        num1=0.0
        den1=0.0
        num2=0.0
        den2=0.0
        for j in range(0,120):
            if max_convulation[T_pool[t_pool_size-1]][j]==0:
                if j >= 60:
                    function(T_pool[t_pool_size - 1], j, 1)
                else:
                    function(T_pool[t_pool_size - 1], j, 0)
                print count
                count += 1
            num1+=output_vector[0][j]*max_convulation[T_pool[t_pool_size-1]][j]
            num2+=output_vector[1][j]*max_convulation[T_pool[t_pool_size-1]][j]
            den1+=pow(max_convulation[T_pool[t_pool_size-1]][j],2)
            den2+=pow(max_convulation[T_pool[t_pool_size-1]][j],2)
        a_k_i[0][t_pool_size-1] = num1 / den1
        a_k_i[1][t_pool_size-1] = num2 / den2
        val=0.0
        for x in range(0,2):
            for y in range(0,120):
                if max_convulation[T_pool[t_pool_size - 1]][y] == 0:
                    if y>=60:
                        function(T_pool[t_pool_size-1],y,1)
                    else:
                        function(T_pool[t_pool_size-1],y,0)
                    print count
                    count += 1
                val+=abs(a_k_i[x][t_pool_size-1])*abs(max_convulation[T_pool[t_pool_size-1]][y])
        influence.append((val,t_pool_size-1))
    influence.sort(key=itemgetter(0))
    x,y=influence[0]
    T_selection.append(T_pool[y])
    for j in range(0,120):
        output_vector[0][j]-=(a_k_i[0][y]*max_convulation[T_pool[y]][j])
        output_vector[1][j]-=(a_k_i[1][y]*max_convulation[T_pool[y]][j])
for i in range(0,len(output_vector)):
    print output_vector[i]
#print time.clock()-start_time

x=np.zeros((1000,120))
#x[0,2]=1
#print x

#max_convulation = [[0 for j in range(120)] for i in range(1000)]
#output_vector = [[0 for j in range(120)] for i in range(2)]

for i in range(1,1001):
    for j in range(1,121):
        #print i," ",j
        if max_convulation[i-1][j-1]==0:
            if j>60:
                function(i, j, 1)
            else:
                function(i, j, 1)
        x[i-1,j-1]=max_convulation[i-1][j-1]

x_t=np.transpose(x)
product=x.dot(x_t)
I=np.identity(1000)
sum=np.add(product,I)
inverse=inv(sum)
temp=inverse.dot(x)
beta=temp.dot(np.transpose(output_vector))

print beta
print time.clock()-start_time









