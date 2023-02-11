import cv2
import time
import mediapipe as mp
import csv
import pandas as pd

lm_store = []

class poseDetector():

    def __init__(self, mode=False, modelComplexity=1, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.modelComplexity = modelComplexity
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComplexity, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList


def array_to_csv(arr):
    headings = ["Landmark", "X Pos", "Y Pos"]
    with open('sample.csv', 'w') as f:
        my_writer = csv.writer(f, delimiter=',')
        my_writer.writerow(headings)
        my_writer.writerows(arr)


def find_range(array, axis, landmark):
    bottom = 0;
    top = 0;
    if axis == 'X' or axis ==  'x':
        axis = 1
    else:
        axis = 2
    for sub_array in array:
        if sub_array[0] == landmark:
            if sub_array[axis] < bottom:
                bottom = sub_array[axis]
            if sub_array[axis] > top:
                top = sub_array[axis]
    return top - bottom


def main():
    cap = cv2.VideoCapture('stock-squat.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)

        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            targets = [0, 13, 14]
            for target in targets:

                print(lmList[target])

                lm_store.append(lmList[target])
                cv2.circle(img, (lmList[target][1], lmList[target][2]), 10, (255, 0, 255), cv2.FILLED)

        c_time = time.time()
        fps = 1 / (c_time - pTime)
        pTime = c_time

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Video", img)
        cv2.waitKey(1)
        print("---------------------------------------")
        print(lm_store)

        array_to_csv(lm_store)
        print("Squat Depth: ",  find_range(lm_store, "Y", 0))


if __name__ == "__main__":
    main()
