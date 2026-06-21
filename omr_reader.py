import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform

def process_custom_omr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

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
        return {"error": "Main question border detect nahi hua. Photo clear nahi hai."}

    warped = four_point_transform(gray, question_box_contour.reshape(4, 2))
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    H, W = thresh.shape
    cw = W / 5.0
    ch = H / 20.0

    answers = {}
    options_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    # 100 Questions loop
    for q_index in range(100):
        c = q_index // 20
        r = q_index % 20

        x_start = c * cw
        y_start = r * ch

        opt_area_x_start = x_start + (cw * 0.28)
        opt_w = (cw * 0.70) / 5.0

        y1 = int(y_start + (ch * 0.15))
        y2 = int(y_start + (ch * 0.85))

        pixels_per_option = []
        for i in range(5):
            x1 = int(opt_area_x_start + (i * opt_w))
            x2 = int(opt_area_x_start + ((i + 1) * opt_w))
            
            box = thresh[y1:y2, x1:x2]
            total_pixels = cv2.countNonZero(box)
            pixels_per_option.append(total_pixels)

        max_pixels = max(pixels_per_option)
        marked_option_idx = pixels_per_option.index(max_pixels)

        if max_pixels > 150:
            sorted_pixels = sorted(pixels_per_option, reverse=True)
            if sorted_pixels[1] > max_pixels * 0.6: 
                answers[str(q_index + 1)] = None # Double marked
            else:
                answers[str(q_index + 1)] = options_map[marked_option_idx]
        else:
            answers[str(q_index + 1)] = None # Blank

    return {
        "ok": True,
        "answers": answers
    }
