import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
image = cv2.imread('ABCDEFGHI.jpg')
grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
preprocessed_digits = []
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
        # (x, y, w, h) = cv2.boundingRect(c)
        # ar = w / float(h)

        # if w >= 15 and h >= 15 and ar >= 0.8 and ar <= 1.5:
        #     questionCnts.append(c)

    if(h >= 100 and h <= 300 and w <= 300):
        print(str(h) + " " + str(w))
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        
        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]
        
        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18,18))
        
        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
        
        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)
print("\n\n\n----------------Contoured Image--------------------")
plt.imshow(image, cmap="gray")
plt.show()
    
inp = np.array(preprocessed_digits)
# MAP DE LETERS 
# letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
# 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
# 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

# MAP DE BALANCED
letters ={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,
10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',
20:'K',21:'l',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',
30:'u',31:'V',32:'W',33:'X',34:'Y',35:'Z',36:'a',37:'b',38:'d',39:'e',
40:'f',41:'g',42:'h',43:'n',44:'q',45:'r',46:'t',47:'அ',48:'ஆ'}

# # #DENSE Model
# new_model = tf.keras.models.load_model('emnist_trained_dense.h5')

#CNN Model
new_model = tf.keras.models.load_model('emnist_trained.h5')

digits_predicted = []


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

for digit in preprocessed_digits:
    # # En caso de usar el model de DENSE
    # prediction = new_model.predict(digit.flatten().reshape(-1, 28*28))  

    # En caso de usar el model de CONVOLUTIONAL
    prediction = new_model.predict(digit.reshape(1, 28, 28, 1))

    print ("---------------------------------------")
    print(prediction)
    
    print ("=========PREDICTION============")
    print("\n\nFinal Output: {}".format(np.argmax(prediction)))
    print("\n\nFinal Output: {}"+str(letters[int(np.argmax(prediction))]))
    
    digits_predicted.append(letters[int(np.argmax(prediction))])
    
    hard_maxed_prediction = np.zeros(prediction.shape)
    hard_maxed_prediction[0][np.argmax(prediction)] = 1
    

    plt.imshow(digit.reshape(28, 28), cmap="gray")
    plt.show()
    # plt.imshow(digit.reshape(28, 28), cmap="gray")
    # plt.show()

print(digits_predicted)
    