import cv2
import numpy as np
#import pyautogui

net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()
fc=0 #face covered index number of file
fm=0 #no mask index number of file
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    #myScreenshot = pyautogui.screenshot()

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #if class_id == 2:
             #   myScreenshot.save('file5.png')

            #if class_id == 1:

             #   myScreenshot.save('file2.png')

            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

                crop=frame[y:y+h, x:x+w]
                if(crop.size>0):
                	#cv2.imshow("cropped",crop)#
                	cv2.waitKey(20)#
                	if class_id == 1:
                		cv2.imwrite('notCov/img'+str(fc)+'.jpg',crop) 
                		fc=fc+1              
                	if class_id == 2:
                		cv2.imwrite('noMask/img'+str(fm)+'.jpg',crop)
                		fm=fm+1   
                


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow('Image', frame)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
