#! /usr/bin/env python3

import rospy
import sys
import cv2
import string
from std_msgs.msg import String
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ClueBoardReader:
    
    def __init__(self):
        
        rospy.init_node('topic_publisher')
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)

        self.reader_NN = models.load_model('/home/fizzer/ros_ws/src/my_controller/models/char_recognition_03.h5')
        self.classes = string.ascii_uppercase + '0123456789'

    def detect_clue(self, img, id):
        print("clue detected")
        preprocessed_img = self.preprocess_image(img, id)
        val_image = self.process_val3(preprocessed_img, id)
        pred_val = self.run_cnn(val_image)

        # Publish clue prediction
        # self.pub_score.publish('Team11,TBD,{},{}'.format(clue_id, val))
        if pred_val is not None:
            self.pub_score.publish('Team11,TBD,{},{}'.format(str(id), pred_val))
            # msg = String(data='Team11,TBD,{},{}'.format(id, pred_val))
            # self.pub_score.publish(msg)
            print(f"Detected sign: {id}, {pred_val}")

        if id == 8:
            # Stop timer
            self.pub_score.publish('Team11,TBD,-1,STOP')

    def preprocess_image(self, img, id):
        if id == 3:
            cropped_img = self.crop_clue_board_light(img, id)
        elif id == 6:
            cropped_img = self.crop_clue_board_light(img, id)
        else:
            cropped_img = self.crop_clue_board_dark(img, id)
        transformed_img = self.perspective_transform(cropped_img)
        return transformed_img

    def crop_clue_board_dark(self, image, id):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the blue color range in HSV
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        
        # Create a mask for the blue regions
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply the mask to the image
        blue_region = cv2.bitwise_and(image, image, mask=mask)

        # Convert the blue region to grayscale for contour detection
        gray_blue = cv2.cvtColor(blue_region, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to get a binary image
        _, thresh = cv2.threshold(gray_blue, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding box of the largest contour (assumed to be the border)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop the image using the bounding box
            cropped_image = image[y:y+h, x:x+w]

            # cv2.imshow("Cropped image dark", cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            # cv2.waitKey(1)

            return cropped_image
        else:
            print("No blue border detected.")

    def crop_clue_board_light(self, image, id):
        x1, y1, x2, y2 = 0, 250, 790, 700
        cropped_img = image[y1:y2, x1:x2]
    
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        # Define the blue color range in HSV
        lower_blue = np.array([100, 100, 150])
        upper_blue = np.array([140, 255, 255])
        
        # Create a mask for the blue regions
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply the mask to the image
        blue_region = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

        # Convert the blue region to grayscale for contour detection
        gray_blue = cv2.cvtColor(blue_region, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image to get a binary image
        _, thresh = cv2.threshold(gray_blue, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding box of the largest contour (assumed to be the border)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop the image using the bounding box
            cropped_image = cropped_img[y:y+h, x:x+w]

            # cv2.imshow("Cropped image light", cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            # cv2.waitKey(1)

            return cropped_image
        else:
            print("No blue border detected.")

    def perspective_transform(self, image):
        """Crops out blue border and straightens the image"""

        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define grey color range
        lower_grey = np.array([0, 0, 50])
        upper_grey = np.array([180, 50, 200])

        # Create a mask to detect the grey regions
        mask = cv2.inRange(hsv, lower_grey, upper_grey)

        # Apply morphological operations to close gaps
        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Draw contours
        # image_with_contours = image.copy()
        # cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        # cv2.imshow("Contours", image_with_contours)
        # cv2.waitKey(1)

        # Process the largest contour (assuming it's the sign)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate the contour to get a quadrilateral
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) == 4:  # Ensure we have 4 corners
                src_pts = np.float32([point[0] for point in approx])

                # Sort points (Top-left, Top-right, Bottom-right, Bottom-left)
                src_pts = sorted(src_pts, key=lambda x: (x[1], x[0]))  # Sort by Y first, then X
                top_pts = sorted(src_pts[:2], key=lambda x: x[0])  # Sort top-left and top-right
                bottom_pts = sorted(src_pts[2:], key=lambda x: x[0])  # Sort bottom-left and bottom-right
                src_pts = np.float32([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]])

                # Define the new (straightened) rectangular coordinates
                width, height = 300, 200  # Adjust output size as needed
                dst_pts = np.float32([
                    [0, 0],
                    [width, 0],
                    [width, height],
                    [0, height]
                ])

                # Compute the perspective transformation matrix
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # Apply perspective warp
                warped = cv2.warpPerspective(image, M, (width, height))

                # Adjust image size
                warped = cv2.resize(warped, (600, 400))

                # Darken clue id 3 and 6
                brightness_decrease = 80
                if id == 3:
                    warped = cv2.subtract(warped, np.full(warped.shape, brightness_decrease, dtype=np.uint8))
                elif id == 6:
                    warped = cv2.subtract(warped, np.full(warped.shape, brightness_decrease, dtype=np.uint8))

                # cv2.imshow("Transformed image", warped)
                # cv2.waitKey(1)

                return warped

            else:
                print(f"Could not detect exactly 4 corners for {image}")
        else:
            print(f"No sign detected for {image}")

    def process_val3(self, img, id):
        img.resize((400, 600, 3))

        # Define the region of interest (ROI) for cropping (if provided)
        x1_val = 0
        y1_val = 230
        x2_val = 600
        y2_val = 400

        img_val = img[y1_val:y2_val, x1_val:x2_val]

        # Decrease brightness by subtracting a value
        brightness_decrease = 70
        darker_image = cv2.subtract(img_val, np.full(img_val.shape, brightness_decrease, dtype=np.uint8))
        if id != 3 and id !=6:
            darker_image = img_val

        gray_val = cv2.cvtColor(darker_image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        blur = cv2.bilateralFilter(gray_val, 9, 75, 75)

        # Create blurred version
        gaussian = cv2.GaussianBlur(blur, (0, 0), 3)

        # Apply unsharp mask
        sharpened = cv2.addWeighted(blur, 2.5, gaussian, -0.8, 1)

        if id == 6:
            # Apply morphological closing to merge broken letters
            kernel = np.ones((10, 10), np.uint8)
            closed = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)

            # Apply adaptive thresholding to handle uneven illumination
            thresh_val = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)
            # thresh_val = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)

        else:
            # Apply adaptive thresholding to handle uneven illumination
            thresh_val = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)

        # Find contours again on the best preprocessed image (without blur)
        contours, _ = cv2.findContours(thresh_val, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get bounding boxes of contours and filter valid ones based on size
        bounding_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10 and h > 50:  # Ignore small noise
                bounding_boxes.append((x, y, w, h))

        # Sort bounding boxes from left to right
        bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])

        # Now calculate median width and height of valid bounding boxes
        widths = [w for _, _, w, h in bounding_boxes]
        heights = [h for _, _, w, h in bounding_boxes]
        median_width = int(np.median(widths)) if widths else 0
        median_height = int(np.median(heights)) if heights else 0

        # Add safety check in case no valid boxes are found
        if median_width == 0 or median_height == 0:
            print("Warning: No valid contours found or median size is zero")
            median_width = max(20, min(widths)) if widths else 20  # Default fallback
            median_height = max(60, min(heights)) if heights else 60  # Default fallback

        # Define size thresholds for merging or splitting
        small_box_threshold = median_width * 0.5  # If width is less than 50% of median, merge it
        large_box_threshold = median_width * 1.8  # If width is more than 180% of median, split it

        # Process bounding boxes
        filtered_boxes = []
        for x, y, w, h in bounding_boxes:
            if w < small_box_threshold:  # Merge small boxes (later step)
                continue
            elif w > large_box_threshold:  # Split wide boxes into two
                filtered_boxes.append((x, y, w // 2, h))
                filtered_boxes.append((x + w // 2, y, w // 2, h))
            else:
                filtered_boxes.append((x, y, w, h))

        # Draw bounding boxes on the image
        if len(darker_image.shape) == 2 or (len(darker_image.shape) == 3 and darker_image.shape[2] == 1):
            # Image is grayscale
            bounding_box_image = cv2.cvtColor(darker_image, cv2.COLOR_GRAY2BGR)
        else:
            # Image is already in color
            bounding_box_image = darker_image.copy()

        for x, y, w, h in filtered_boxes:
            cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display results
        cv2.imshow("Bounding boxes", cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
        
        output_size = (32, 32)
        value_images = []

        # Process value characters
        for (x, y, w, h) in filtered_boxes:
            char_img_val = thresh_val[y:y+h, x:x+w]
            char_img_val = cv2.resize(char_img_val, output_size)  # Resize to CNN input size
            char_img_val = np.stack((char_img_val,)*3, axis=-1)  # Convert grayscale to 3-channel (RGB)
            value_images.append(char_img_val)

        return value_images

    def run_cnn(self, value_images):
        
        detected_values = []

        # Process value characters
        for char_img in value_images:
            char_img = char_img / 255.0  # Normalize
            char_img = np.expand_dims(char_img, axis=0)  # Add batch dimension
            
            # Run CNN model prediction
            prediction = self.reader_NN.predict(char_img)
            class_index = np.argmax(prediction)  # Get highest probability class
            class_labels = self.classes
            detected_values.append(class_labels[class_index])

        # Convert to string
        detected_values_str = ''.join(detected_values)

        print(f"Detected Values: {detected_values_str}")
        
        return detected_values_str
    
    def start_timer(self):
        self.pub_score.publish('Team11,TBD,0,START')

    def loop(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        reader = ClueBoardReader()
        reader.loop()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()