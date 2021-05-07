import cv2

# Load the classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# video from the webcam.
cap = cv2.VideoCapture(0)


#dHash algorithm
def dHash(img):
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dhash_str=''
    for i in range(8):                #The previous pixel in each line is greater than the next pixel is 1, and the opposite is 0, generating a hash
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                dhash_str = dhash_str+'1'
            else:
                dhash_str = dhash_str+'0'
    return dhash_str

def cmpHash(hash1,hash2):              #Hash value comparison
    n=0
    if len(hash1)!=len(hash2):         #If the hash length is different, it will return -1 to indicate an error in parameter transmission
        return -1
    for i in range(len(hash1)):        #judgment
        if hash1[i]!=hash2[i]:         #If they are not equal, n counts +1, and n is the similarity
            n=n+1
    return n

#judgment
def judgment(img):        
    cv2.imwrite('F:/test2.jpg', img) 
    img1 = cv2.imread('F:/test2.jpg')
    img2 = cv2.imread('F:/test3.png')
    hash1= dHash(img1)
    hash2= dHash(img2)
    n=cmpHash(hash1,hash2)
    print('dHash algorithm similarity：',n)
    if n<30:
        print('Success')
    else:
        print('Fault')

#camera
def cam():    
    while True:
        # Read the frame
        _, img = cap.read()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(25, 25))
        

        # Draw the box of the face part
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 1)
            judgment(img)
         
        # Show results
        cv2.imshow('img', img)
       
        # Press ESC to end the program execution
        k = cv2.waitKey(30) & 0xff
        if k==27:           
            break
    
    # Release the VideoCapture object     
    cap.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    cam()    
