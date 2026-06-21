import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform

def process_custom_omr(image):
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Outer border dhundhna
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    question_box_contour = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                question_box_contour = approx
                break

    if question_box_contour is None:
        return {"error": "Main question border detect nahi hua. Photo saaf nahi hai."}

    # Image ko seedha (Deskew) karna
    warped = four_point_transform(gray, question_box_contour.reshape(4, 2))
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Bubbles (Circles) dhundhna
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    question_bubbles = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 15 and h >= 15 and ar >= 0.8 and ar <= 1.2:
            question_bubbles.append(c)

    return {
        "ok": True,
        "message": "Image successfully cropped and deskewed.",
        "bubbles_found": len(question_bubbles),
        "expected_bubbles": 500
    }
