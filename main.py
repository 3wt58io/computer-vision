import cv2
import numpy as np
import time


# Set ROI to cover the signal lens (x, y, width, height). Other modes are
# available by setting TARGET_COLOR to "red" or "black".
ROI = (100, 100, 200, 200)
TARGET_COLOR = "red_light"
ALERT_AFTER_SECONDS = 3.0
MIN_COLOR_PERCENT = 2.0
MIN_RED_LIGHT_PERCENT = 2.0
MIN_RED_PERCENT = 20.0
RED_LIGHT_MIN_SATURATION = 150
RED_LIGHT_MIN_BRIGHTNESS = 180


def create_color_masks(frame):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_hue_mask = cv2.bitwise_or(
        cv2.inRange(hsv_image, np.array([0, 100, 80]), np.array([10, 255, 255])),
        cv2.inRange(hsv_image, np.array([170, 100, 80]), np.array([180, 255, 255])),
    )
    red_light_hue_mask = cv2.bitwise_or(
        cv2.inRange(
            hsv_image,
            np.array([0, RED_LIGHT_MIN_SATURATION, RED_LIGHT_MIN_BRIGHTNESS]),
            np.array([8, 255, 255]),
        ),
        cv2.inRange(
            hsv_image,
            np.array([172, RED_LIGHT_MIN_SATURATION, RED_LIGHT_MIN_BRIGHTNESS]),
            np.array([180, 255, 255]),
        ),
    )
    blue_channel, green_channel, red_channel = cv2.split(frame)
    red_dominance_mask = cv2.bitwise_and(
        cv2.compare(red_channel, green_channel, cv2.CMP_GT),
        cv2.compare(red_channel, blue_channel, cv2.CMP_GT),
    )
    red_light_mask = cv2.bitwise_and(red_light_hue_mask, red_dominance_mask)
    black_mask = cv2.inRange(
        hsv_image, np.array([0, 0, 0]), np.array([180, 255, 60])
    )

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_hue_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_light_mask = cv2.morphologyEx(
        red_light_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    return {"red": red_mask, "red_light": red_light_mask, "black": black_mask}


def detect_red_and_black_objects(frame, color_masks=None):
    if color_masks is None:
        color_masks = create_color_masks(frame)

    for mask, label, color in (
        (color_masks["red"], "Red", (0, 0, 255)),
        (color_masks["red_light"], "Red light", (0, 0, 255)),
        (color_masks["black"], "Black", (255, 255, 255)),
    ):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            min_area = 10 if label == "Red light" else 500
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

    combined_mask = cv2.bitwise_or(
        color_masks["red_light"],
        cv2.bitwise_or(color_masks["red"], color_masks["black"]),
    )
    return frame, combined_mask


cap = cv2.VideoCapture(0)
color_present_since = None
alert_printed = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check camera connection.")
            break

        x, y, width, height = ROI
        frame_height, frame_width = frame.shape[:2]
        if (
            width <= 0
            or height <= 0
            or x < 0
            or y < 0
            or x + width > frame_width
            or y + height > frame_height
        ):
            raise ValueError(
                f"ROI {ROI} is outside the camera frame "
                f"({frame_width}x{frame_height}); adjust ROI near the top of main.py."
            )
        if TARGET_COLOR not in ("red_light", "red", "black"):
            raise ValueError(
                'TARGET_COLOR must be "red_light", "red", or "black".'
            )
        if ALERT_AFTER_SECONDS < 0:
            raise ValueError("ALERT_AFTER_SECONDS must not be negative.")
        if not 0 <= MIN_COLOR_PERCENT <= 100:
            raise ValueError("MIN_COLOR_PERCENT must be between 0 and 100.")
        if not 0 <= MIN_RED_LIGHT_PERCENT <= 100:
            raise ValueError("MIN_RED_LIGHT_PERCENT must be between 0 and 100.")
        if not 0 <= MIN_RED_PERCENT <= 100:
            raise ValueError("MIN_RED_PERCENT must be between 0 and 100.")

        color_masks = create_color_masks(frame)
        roi_mask = color_masks[TARGET_COLOR][y : y + height, x : x + width]
        color_percent = cv2.countNonZero(roi_mask) / (width * height) * 100
        red_light_percent = color_percent
        red_percent = color_percent
        if TARGET_COLOR == "red_light":
            red_roi_mask = color_masks["red"][y : y + height, x : x + width]
            red_percent = cv2.countNonZero(red_roi_mask) / (width * height) * 100
            meets_threshold = (
                red_light_percent >= MIN_RED_LIGHT_PERCENT
                and red_percent >= MIN_RED_PERCENT
            )
        else:
            meets_threshold = color_percent >= MIN_COLOR_PERCENT
        now = time.monotonic()

        if meets_threshold:
            if color_present_since is None:
                color_present_since = now
            elapsed = now - color_present_since
            if elapsed >= ALERT_AFTER_SECONDS and not alert_printed:
                if TARGET_COLOR == "red_light":
                    print(
                        f"ALERT: ROI contains at least {MIN_RED_LIGHT_PERCENT:.0f}% "
                        f"red light and {MIN_RED_PERCENT:.0f}% red for "
                        f"{ALERT_AFTER_SECONDS:g} seconds."
                    )
                else:
                    print(
                        f"ALERT: {TARGET_COLOR} has covered at least "
                        f"{MIN_COLOR_PERCENT:.0f}% of the ROI for "
                        f"{ALERT_AFTER_SECONDS:g} seconds."
                    )
                alert_printed = True
        else:
            color_present_since = None
            alert_printed = False
            elapsed = 0

        frame, mask = detect_red_and_black_objects(frame, color_masks)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        if TARGET_COLOR == "red_light":
            status = (
                f"Red light: {red_light_percent:.1f}/{MIN_RED_LIGHT_PERCENT:.0f}%  "
                f"Red: {red_percent:.1f}/{MIN_RED_PERCENT:.0f}%"
            )
        else:
            status = f"{TARGET_COLOR}: {color_percent:.1f}%"
        if color_present_since is not None:
            status += f" ({elapsed:.1f}/{ALERT_AFTER_SECONDS:g}s)"
        cv2.putText(
            frame, status, (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        cv2.imshow("Camera", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
