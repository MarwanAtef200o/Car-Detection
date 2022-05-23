import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


#load yolo weights and config
weights_path= os.path.join("yolo","yolov3.weights")
config_path= os.path.join("yolo","yolov3.cfg")

#load the neural net in cv2
net= cv2.dnn.readNetFromDarknet(config_path, weights_path)

#get net's Layers Names
names= net.getLayerNames()



def framePipeline(img):
    (H, W)= img.shape[:2]

    #yolo_82,yolo_94,yolo_106 are the layers to produce the output (prediction layers are unconnected)
    layers_names= [names[i - 1] for i in net.getUnconnectedOutLayers()]

    #run the inference on the test image
    #blob is used as an input in the neural net to make prediction
    blob= cv2.dnn.blobFromImage(img, 1/255.0, (416,416), crop= False, swapRB= False)
    net.setInput(blob)
    #output of the neural network (the output of the 3 yolo layers)
    layers_output= net.forward(layers_names)

    boxes= []
    confidences= []
    classIDs= []

    for output in layers_output:
        for detection in output:
            # scores of the classes from index 5 till the end of the classes
            scores= detection[5:]
            #choose the maximum class score
            classID=np.argmax(scores)
            #get the confidence of the max score class (confidence that a certain object is in the img)
            confidence= scores[classID]

            if(confidence > 0.85):
                #indecies 0 to 4 tells me whether an object is in img (Pc). width,height,center of the box
                box= detection[:4] * np.array([W, H, W, H])
                bx, by, bw, bh = box.astype("int")

                #coords of the top left corner of the box
                x= int(bx - (bw / 2))
                y= int(by - (bh / 2))

                #append the box, the confidence and classID
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #0.8 is threshold of bounding box actually encloses an object,
    #if there are boxes bounding the same object, then choose the higher confidence score box (NMS)
    idxs= cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.8)
    #give indecies to the chosen box

    #read the labels file
    labels= open("yolo/coco.names").read().strip().split("\n")

    #plot the bounding boxes in the image
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x,y)= [boxes[i][0], boxes[i][1]]
            (w,h)= [boxes[i][2], boxes[i][3]]

            cv2.rectangle(img, (x,y), (x+w, y+h), (0,164,164), 2)
            cv2.putText(img, "{}: {}".format(labels[classIDs[i]], confidences[i]), (x, y-5), cv2.FONT_ITALIC, 0.5, (0,164,164), 2)


    outputImg=img
    imgStack = stackImages(0.4, ([img1],
                                 [outputImg]))
    return imgStack


path= sys.argv[2]

if sys.argv[1] == "img":

    img = cv2.imread(path)
    img1 = cv2.imread(path)
    img, img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR),cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    plt.imshow(framePipeline(img))
    plt.show()


elif sys.argv[1] == "vid":
    video = cv2.VideoCapture(path)
    frameWidth = int(video.get(3))
    frameHeight = int(video.get(4))

    while video.isOpened():
        ret1, img1 = video.read()
        ret, frame = video.read()

        if ret == True:
            #img1= frame
            frame = framePipeline(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('s'): #press s to stop the running video
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()