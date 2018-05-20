"""Webcam utilities."""

import cv2

import GFX

KEY_ESC = 27

def capture():
    cap = cv2.VideoCapture(0)

    while (True):
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        cv2.imshow('frame', rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return GFX.Image(frame)
    # cap.release()
    # cv2.destroyAllWindows()

    # vc = cv2.VideoCapture(0)
    # if vc.isOpened():
    #     _, frame = vc.read()
    #     return gfx.Image(frame)


def display():
    vc = cv2.VideoCapture(0)
    key = 0
    success = True

    face_detector = GFX.FaceDetector()

    while success and key != KEY_ESC:
        success, frame = vc.read()
        face_detector.show(GFX.Image(frame), wait=False)
        key = cv2.waitKey(20)
