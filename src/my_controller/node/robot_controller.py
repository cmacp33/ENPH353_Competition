#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from clue_detection import ClueBoardReader

class Robot:

    def __init__(self):

	    # clueboard reader initialization
        self.clue_reader = ClueBoardReader()
        self.clue_reader.start_timer()
        
	    # init ROS nodes
        rospy.init_node('topic_publisher')
        self.pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)
        self.sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.cam_cb, queue_size=3)

	    # init bridge and twist
        self.bridge = CvBridge()
        self.move = Twist()

	    # set rate
        self.rate = rospy.Rate(1)
        
	    # init controller params
        self.step = np.zeros(11)
        self.prev_img = None
        self.count = 0
        self.prev_error = 0
        self.middle_offset = 0

	    # mode flags
        self.yoda_follow = False
        self.car_follow = False
        self.sign_follow = False

	    # spawn car in start position
        self.start_position = [5.5, 2.5, 0.2, 0.0, 0.0, -0.70710678, 0.70710678]
        self.spawn_position(self.start_position)

	    # init clue board images
        self.cb1 = None
        self.cb2 = None
        self.cb3 = None
        self.cb4 = None
        self.cb5 = None
        self.cb6 = None
        self.cb7 = None
        self.cb8 = None

	    # color thresholds
        self.lower_blue = np.array([0, 0, 95])
        self.upper_blue = np.array([10, 10, 120])
        self.lower_red = np.array([0, 0, 240])
        self.upper_red = np.array([10, 10, 255])
        self.lower_white = np.array([240, 240, 240])
        self.upper_white = np.array([255, 255, 255])
        self.lower_road = np.array([80, 80, 80])
        self.upper_road = np.array([142, 142, 142])
        self.lower_pink = np.array([230, 0, 230])
        self.upper_pink = np.array([255, 30, 255])
        self.lower_yoda = np.array([38, 38, 38])
        self.upper_yoda = np.array([45, 45, 45])
        self.lower_car = np.array([40, 40, 100])
        self.upper_car = np.array([50, 50, 120])
        self.lower_sign = np.array([100, 30, 30])
        self.upper_sign = np.array([255, 100, 100])
    

    # spawn car in start position
    def spawn_position(self, position):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            msg = ModelState()
            msg.model_name = 'B1'
            msg.pose.position.x = position[0]
            msg.pose.position.y = position[1]
            msg.pose.position.z = position[2]
            msg.pose.orientation.x = position[3]
            msg.pose.orientation.y = position[4]
            msg.pose.orientation.z = position[5]
            msg.pose.orientation.w = position[6]

            resp = set_state(msg)
            rospy.loginfo("Robot teleported successfully!")

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)


    # returns true if color threshold is above the target area (m^2)
    def contour_detect(self, img, lower_thresh, upper_thresh, targ_area):

        area = 0
        mask = cv2.inRange(img, lower_thresh, upper_thresh)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            #print(f"Contour Area: {area}")

        if (area >= targ_area):
            # print("Condition met")
            return True

        return False


    # returns true if image difference (motion) is above target area
    def motion_detect(self, img, targ_area):

        count = self.count
        if self.count < 10:
            self.count = self.count + 1
            return
        
        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.prev_img, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        _, self.thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            # print(f"Motion detect area {area}")
            if area > targ_area:
                self.count = count
                return True   
            return False
        

    # moves car 
    def move(self, lin_spd, ang_spd, time):

        start_time = rospy.Time.now().to_sec()

        while rospy.Time.now().to_sec() - start_time < time:
            self.move.angular.z = ang_spd
            self.move.linear.x = lin_spd
            self.pub.publish(self.move)

        self.move.angular.z = 0
        self.move.linear.x = 0
        self.pub.publish(self.move)


    # stops car
    def stop(self):

        self.move.angular.z = 0
        self.move.linear.x = 0
        self.pub.publish(self.move)


    # follows path, signs, yoda, car, etc
    def pathfollow(self, cv_image):

        height, width = cv_image.shape[:2]
        cv_img_center = width // 2
        set_speed = 1

        if self.sign_follow:
            target, target_height = self.center_2D(cv_image, self.lower_blue, self.upper_blue)
            P = 0.005
            D = 0.0001   
        elif self.car_follow:
            target, target_height = self.center_2D(cv_image, self.lower_car, self.upper_car)
            P = 0.005
            D = 0.0001   
        elif self.yoda_follow:
            target, target_height = self.center_2D(cv_image, self.lower_yoda, self.upper_yoda)
            P = 0.005
            D = 0.0001
        else: 
            target = self.get_line_center(cv_image, cv_img_center)
            P = 0.025
            D = 0.005
        
        if target == -404:
            self.move.linear.x = 0.0
            self.move.angular.z = 0.0
            return
        
        error = target - cv_img_center
        derivative = (error - self.prev_error)
        self.prev_error = error
        
        speed_adjustment = P * abs(error) + D * derivative
        
        if error > 5:
            self.move.angular.z = -(set_speed + speed_adjustment)
        elif error < -5:
            self.move.angular.z = (set_speed + speed_adjustment)
        else:
            self.move.angular.z = 0.0
        
        if self.car_follow:
            self.move.linear.x = 0.75
        elif self.yoda_follow:
            if target_height > 700:
                self.move.linear.x = 0.0
            elif abs(error) < 4:
                self.move.linear.x = 1
            else:
                self.move.linear.x = 1
        else:
            if abs(error) < 20:
                self.move.linear.x = 1
            else:
                self.move.linear.x = 1.5
        
        self.pub.publish(self.move)
    

    # returns image with background black and path white
    def find_path(self, cv_img):
        
        cropped_img = cv_img[-50:,:]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        thresh_inverted = cv2.bitwise_not(thresh)
        contours, _ = cv2.findContours(thresh_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(cropped_img, [max_contour], -1, (255, 0, 0), 3)
            self.show_img = cropped_img
        return thresh_inverted
    

    # returns the x coord for the path center of cv_img
    def get_line_center(self, cv_img, img_center):

        new_cv_img = self.find_path(cv_img)
        height, width = new_cv_img.shape[:2]
        contours, hierarchy = cv2.findContours(new_cv_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        if len(contours) == 0:
            return -404
        
        x_contours = []
        target_y = height - 50
        tolerance = 50
            
        for point in contour:
            x, y = point[0]
            if abs(y - target_y) <= tolerance:
                x_contours.append(x)
        
        if len(x_contours) == 0:
            return -404
        
        left = np.min(x_contours)
        right = np.max(x_contours)
        middle = (left + right) // 2 - self.middle_offset

        radius = 10
        color = (255, 0, 255)
        line_thickness = -1

        cv2.circle(self.show_img, (middle, target_y), radius, color, line_thickness)
        
        return middle

	
    # returns the x and y coords for the center of the largest contour (from lower_thresh to upper_thresh) in img
    def center_2D(self, img, lower_thresh, upper_thresh):

        new_img = cv2.inRange(img, lower_thresh, upper_thresh)
        h, w = new_img.shape[:2]
        contours, _ = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            #print(f"2D area {area}")
            if area > 100:
                image = cv2.drawContours(img.copy(), [largest_contour], -1, (0, 255, 0), 2)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.show_img = cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
                    return cx, cy
        else:
            self.show_img = img

        return -404, h


    # keeps ROS alive
    def loop(self):
      rospy.spin()


    # displays camera feed and activates the clue reader when a picture is taken of the clueboard
    def cam_cb(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            self.show_img = cv_image
            self.img_process(cv_image)
            self.prev_img = cv_image

            cv2.imshow("Camera Feed", cv_image)

            for i in range(1, 9):
                cb = getattr(self, f"cb{i}", None)
                if cb is not None:
                    self.clue_reader.detect_clue(cb, i)
                    setattr(self, f"cb{i}", None)

            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)


    # marks step as done
    def complete_step(self, step_num):
        self.step[step_num] = 1
        self.count = 0
        print(f"Step {step_num} Complete")


    # runs all steps in order
    def img_process(self, img):
        for i in range(1, 12):
            if self.step[i] == 0:
                step_fn = getattr(self, f"step{i}", None)
                if step_fn:
                    step_fn(img)
                return


    # take first clueboard pic and go to red stop line
    def step1(self, img):
        if self.contour_detect(img, self.lower_blue, self.upper_blue, 10000):
            self.cb1 = img
        elif self.contour_detect(img, self.lower_red, self.upper_red, 16000):
            self.complete_step(1)
            self.stop()
        else:
            self.pathfollow(img)
    

    # detect when pedestrain moves and road is safe to cross
    def step2(self, img):
        if self.motion_detect(img, 600):
            self.complete_step(2)

	
    # take clueboard 2 & 3 pics and move up to truck loop
    def step3(self, img):
        if self.contour_detect(img, self.lower_blue, self.upper_blue, 10000):
            self.cb2 = img
        elif self.contour_detect(img, self.lower_sign, self.upper_sign, 15000):
            self.cb3 = img
        elif self.contour_detect(img, self.lower_road, self.upper_road, 215000):
            self.stop()
            self.complete_step(3)
        else:
            self.pathfollow(img)


    # wait for truck to pass
    def step4(self, img):
        if self.motion_detect(img, 1000):
            self.complete_step(4)


    # go through loop, exit loop and take picture of clueboard 4
    def step5(self, img):
        if self.count == 0:
            self.move(1,-1.5,1)
        elif self.contour_detect(img, self.lower_blue, self.upper_blue, 10000):
            self.cb4 = img
        elif self.count < 400:
            self.pathfollow(img)
        else:
            self.complete_step(5)
        self.count += 1


    # take pictures of clueboard 6 and 5, road follow until magenta line
    def step6(self, img):
        if self.contour_detect(img, self.lower_blue, self.upper_blue, 10000):
            self.cb5 = img
        elif self.contour_detect(img, self.lower_sign, self.upper_sign, 15000):
            self.cb6 = img
        else:
            self.pathfollow(img)
            if self.contour_detect(img, self.lower_pink, self.upper_pink, 8000):
                self.stop
                self.complete_step(6)
        

    # wait until yoda is in sight
    def step7(self, img):
        if self.contour_detect(img, self.lower_yoda, self.upper_yoda, 250):
            self.complete_step(7)
    

    # follow yoda until red car is in sight, then follow red car
    def step8(self, img):
        if self.contour_detect(img, self.lower_car, self.upper_car, 100):
            self.car_follow = True
            self.pathfollow(img)
            if self.contour_detect(img, self.lower_car, self.upper_car, 4000):
                self.move(0,3.14,0.75)
                self.car_follow = False
                self.yoda_follow = False
                self.complete_step(8)
        else:
            self.yoda_follow = True
            self.pathfollow(img)


    # turn 90deg left at the red car and take a picture of clueboard 7 
    def step9(self, img):
        if self.count == 0:
            self.move(0,2,0.5)
            self.count += 1
        else:
            self.cb7 = img
            self.complete_step(7)


    # follow path until clueboard is in sight
    def step10(self, img):
        if self.contour_detect(img, self.lower_blue, self.upper_blue, 750):
            self.complete_step(10)
        else:
            self.pathfollow(img)
    

    # follow the clueboard and take the last picture, then reset
    def step11(self, img):
        if self.contour_detect(img, self.lower_blue, self.upper_blue, 10000):
            self.cb8 = img
            self.sign_follow = False
            self.complete_step(11)
            self.step = np.zeros(11)
            print("Done!")
            self.spawn_position(self.start_position)
        else:
            self.sign_follow = True
            self.pathfollow(img)


if __name__ == '__main__':
    try:
        rb = Robot()
        rb.loop()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
