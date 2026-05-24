import cv2
import numpy as np

cap = cv2.VideoCapture(0)

pixels_per_cm = 37

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(blurred, 50, 100)

    contours, _ = cv2.findContours(
        edged.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        if cv2.contourArea(cnt) < 1000:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        width = rect[1][0] / pixels_per_cm
        height = rect[1][1] / pixels_per_cm

        text = "W:{:.1f}cm H:{:.1f}cm".format(width, height)

        cv2.putText(
            frame,
            text,
            (int(rect[0][0]), int(rect[0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    cv2.imshow("Object Dimension Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
