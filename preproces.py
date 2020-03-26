import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
image = cv2.imread('20200326_002147.jpg')
print(image)
grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

print(grey)
ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
preprocessed_digits = []
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
        # (x, y, w, h) = cv2.boundingRect(c)
        # ar = w / float(h)

        # if w >= 15 and h >= 15 and ar >= 0.8 and ar <= 1.5:
        #     questionCnts.append(c)

    if(h >= 100):
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

new_model = tf.keras.models.load_model('num_reader_medium.model')

digits_predicted = []
for digit in preprocessed_digits:

    prediction = new_model.predict(digit.reshape(1, 28, 28, 1))  
    
    print ("\n\n---------------------------------------\n\n")
    print ("=========PREDICTION============ \n\n")
    print("\n\nFinal Output: {}".format(np.argmax(prediction)))
    digits_predicted.append(np.argmax(prediction))

    plt.imshow(digit.reshape(28, 28), cmap="gray")
    plt.show()
    
    print ("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))
    
    hard_maxed_prediction = np.zeros(prediction.shape)
    hard_maxed_prediction[0][np.argmax(prediction)] = 1
    print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
    print ("\n\n---------------------------------------\n\n")

    # plt.imshow(digit.reshape(28, 28), cmap="gray")
    # plt.show()

print(digits_predicted)
    