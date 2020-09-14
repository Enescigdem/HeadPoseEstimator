
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from math import cos, sin
import torch.utils.data as utils
import cv2
import numpy as np
import torchvision.transforms.functional as F
import pandas as pd
from torch.autograd import Variable
from PIL import Image
import copy


def draw_pose(img, yaw, pitch, roll, tdx=None, tdy=None, size=50):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


####Little myNetwork

resnetmodel = models.resnet34(pretrained=True)


class myNetwork(nn.Module):
    def __init__(self):
        super(myNetwork, self).__init__()

        self.classifier = nn.Sequential(

            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        x = x.view(-1, 512)
        x = self.classifier(x)

        return x


def draw_faceboxes_poses(detected, input_img, faces, ad, img_size, img_w, img_h, model):
    # loop over the detections
    if detected.shape[2] > 0:
        for i in range(0, detected.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detected[0, 0, i, 2]
            # filter out weak detections
            if confidence > 0.6:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                (h0, w0) = input_img.shape[:2]
                box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
                (startX, startY, endX, endY) = box.astype("int")
                # print((startX, startY, endX, endY))
                x1 = startX
                y1 = startY
                w = endX - startX
                h = endY - startY

                x2 = x1 + w
                y2 = y1 + h

                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)

                faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                # faces[i,:,:,:] = cv2.normalize(faces[i,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                face = np.expand_dims(faces[i, :, :, :], axis=0)
                im = input_img[yw1:yw2 + 1, xw1:xw2 + 1, :]

                p_result = predict(model, im)
                print(p_result.data)

                face = face.squeeze()
                img = draw_pose(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], p_result[0][0], p_result[0][1], p_result[0][2])

                input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img
                cv2.rectangle(input_img, (xw2, yw2), (xw1, yw1), (0, 255, 0), 2)
                cv2.imshow("result", input_img)
    else:
        cv2.imshow("result", input_img)

    return input_img


def predict(model, inputs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    inputf = preprocess(Image.fromarray(inputs)).float().to(device)
    model = model.to(device)
    p_result = model(inputf.unsqueeze(0))
    return p_result


def main():
    img_size = 224
    ad = 0.6
    img_idx = 0
    skip_frame = 1

    protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detector",
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # capture video
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024 * 1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768 * 1)

    print('Starting ...')
    detected_pre = np.empty((1, 1, 1))

    #######OUr model is initiating and loading
    resnetmodel = models.resnet34(pretrained=True)
    resnetmodel.fc = myNetwork()
    device = torch.device("cuda")
    resnetmodel = torch.load("resnet3420epochmae.pth")

    while True:
        # get video frame
        ret, input_img = cap.read()

        img_idx = img_idx + 1
        img_h, img_w, _ = np.shape(input_img)

        if img_idx == 1 or img_idx % skip_frame == 0:

            # detect faces using LBP detector
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

            blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detected = net.forward()

            if detected_pre.shape[2] > 0 and detected.shape[2] == 0:
                detected = detected_pre

            faces = np.empty((detected.shape[2], img_size, img_size, 3))

            input_img = draw_faceboxes_poses(detected, input_img, faces, ad, img_size, img_w, img_h, resnetmodel)

        else:
            input_img = draw_faceboxes_poses(detected, input_img, faces, ad, img_size, img_w, img_h, resnetmodel)

        if detected.shape[2] > detected_pre.shape[2] or img_idx % (skip_frame * 3) == 0:
            detected_pre = detected

        key = cv2.waitKey(1)


if __name__ == '__main__':
    main()











