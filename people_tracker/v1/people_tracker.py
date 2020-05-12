import math

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2
import queue
from bus_state import BusState

from deepsort_tracking.deepsort import *

class PeopleTracker:
    """ Detects and Tracks People within the video stream and detects which people have paid

    TALK MORE ABOUT IT HERE
    """
    class Passenger(TrackableObject):
        has_paid = False
        is_valid = False
        stop_entered = 0
        stop_exit = 0
        time_enter = 0
        time_exit = 0

    USE_ROI_MASKING = False
    #curr width=281, height=500
    #Line to determine if person is a passenger or not, make sure it's across the entire frame
    #         BL        Top        Bottom
    PASSENGER_LINE = [(410, 0), (-200, 500)]

    #The side of the line to determine if passenger, can be 0 or 1
    PASSENGER_SIDE = 0

    # Note: These vertices are based on a 500x500 image, if dimensions change, change the ROI
    # U=Upper:  B=Bottom:  R=Right:  L=Left
    #                              UL       UR        BR          BL
    ROI_MASK_VERTICES = np.array([[0, 0], [300, 0], [300, 500], [0, 500]], np.int32)
    USE_HIGH_RES = False

    rects = [] # stores the bounding boxes saved by the tracker used by the centroid tracker
    status = ""
    is_video_file_stream = False
    payment_id_queue = None
    total_down = 0  # Remove this once not needed
    total_up = 0    # Remove this once not needed

    CONFIDENCE = 0.9  # The confidence needed from the detector of object classifications
    SKIP_FRAMES = 10  # How often to run the person detection algorithm

    classPath = "./faster_rcnn_inception_v2_coco/object_detection_classes_coco.txt"
    CLASSIFICATIONS = open(classPath).read().strip().split("\n") # classes stored in txt file

    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    def __init__(self, payment_id_queue, terminals, line_div, pbtxt, model, use_gpu, video_file=None, output_file=None):
        """Initializer

        Args:
            payment_id_queue: stores triggered payments
            terminals: array of terminals within the videos Field Of View
            line_div: when users cross this line they become passengers
            pbtxt: The Model: Holds the network of nodes, each representing a operation
            model: The Model: In binary
            video_file: The input video file, if empty use webcam
            output_file: Stores the processed video
        """

        # initialize bus state class
        self.bus_state = BusState()

        print("video file: ", video_file)
        print("output: ", output_file)
        self.payment_id_queue = payment_id_queue
        self.terminals = terminals
        self.line_div = line_div

        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromTensorflow(model, pbtxt)

        # check if we are going to use GPU
        if use_gpu:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.video_stream = self.load_input_feed(video_file)
        self.output_file = output_file

        # initialize output window
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

        # initialize OpenCV's special multi-object tracker
        self.trackers = cv2.MultiTracker_create()

        self.deepsort = deepsort_rbc()

    def load_input_feed(self, video_file=False):
        """If video_file=true, stream the video from the video, else stream video from cam

        Args:
            video_file: the name of the input video file

        Returns:
            VideoStream if video_file provided else cv2.VideoCapture object(VIDEO CAPTURE NOT STARTED)
        """
        if video_file:
            self.is_video_file_stream = True
            print("[INFO] opening video file...")
            video_stream = cv2.VideoCapture(video_file)
        else:
            self.is_video_file_stream = False
            print("[INFO] init video stream...")
            video_stream = VideoStream(src=0)
        return video_stream

    def check_centroid_side(self, foot_point):
        if foot_point is None:
            print("centroid none?")
            return -1
        x1 = self.PASSENGER_LINE[0][0]
        x2 = self.PASSENGER_LINE[1][0]
        y1 = self.PASSENGER_LINE[0][1]
        y2 = self.PASSENGER_LINE[1][1]

        xA = foot_point[0]
        yA = foot_point[1]

        v1 = (x2 - x1, y2 - y1)  # Vector 1
        v2 = (x2 - xA, y2 - yA)  # Vector 1
        xp = v1[0] * v2[1] - v1[1] * v2[0]  # Cross product
        if xp > 0:
            return 0
        else:
            return 1
        #elif xp <= 0:
        #    return 1
        #    #print('on the other')
        #else:
        #    print('on the same line!')
        #return 0

    def run(self):
        """Runs the PersonTracker detection

        Returns:
            None
        """

        if not self.is_video_file_stream:
            print("[INFO] start video stream...")
            self.video_stream = self.video_stream.start()
            time.sleep(2.0)  # Delay to allow the camera and video stream to load up

        # initialize the video writer (we'll instantiate later if need be)
        writer = None

        # initialize the original_frame dimensions (we'll set them as soon as we read
        # the first original_frame from the video)
        frame_width = None
        frame_height = None

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a tracked person
        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        tracked_people = {}

        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        total_frames = 0

        # start the frames per second throughput estimator
        fps = FPS().start()

        # loop over frames from the video stream
        while True:
            # grab the next original_frame and handle if we are reading from either
            # VideoCapture or VideoStream
            original_frame = self.video_stream.read()

            # Get the first index of the original_frame (the numpy array)
            if self.is_video_file_stream:
                original_frame = original_frame[1]

            # if we are viewing a video and we did not grab a original_frame then we
            # have reached the end of the video
            if self.is_video_file_stream and original_frame is None:
                print("[INFO] END OF INPUT VIDEO")
                break

            # resize the original_frame to have a maximum frame_width of 500 pixels (the
            # less data we have, the faster we can process it), then convert
            # the original_frame from BGR to RGB for dlib
            original_frame = imutils.resize(original_frame, width=500, inter=cv2.INTER_AREA)
            #print("width:", original_frame.shape[0], "    height", original_frame.shape[1])
            processed_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

            # if the original_frame dimensions are empty, set them
            if frame_width is None or frame_height is None:
                (frame_height, frame_width) = original_frame.shape[:2]

            line_location = int(float(frame_height) / self.line_div)

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if self.output_file is not None and writer is None:
                print('[info] file writer initialized')
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                print('writer write to: ', self.output_file)
                writer = cv2.VideoWriter(self.output_file, fourcc, 20.0,
                                         (frame_width, frame_height), True)

            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            self.status = "Waiting"
            self.rects = []

            if self.USE_ROI_MASKING:
                processed_frame = self.region_of_interest(original_frame, processed_frame, self.ROI_MASK_VERTICES)

            boxes, confidences = self.object_detection(original_frame, processed_frame)
            self.deepsort_tracking(boxes, confidences, original_frame)

            #cv2.line(original_frame, (0, line_location), (frame_width,line_location), (0, 255, 255), 2)
            cv2.line(original_frame, self.PASSENGER_LINE[0], self.PASSENGER_LINE[1], (0, 0, 0), 1)

            # draw payment terminals
            for termID, (x, y) in enumerate(self.terminals):
                cv2.circle(original_frame, (x, y), 4, (255, 0, 255), -1)
                cv2.putText(original_frame, f"term {termID}", (x - 20, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


            #TYLER TODO, detect unpaid tracked objects and see if they had paid

            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            # added deregisteredID to update bus_state correctly
            objects, deregisteredIDs = ct.update(self.rects)
            # get the deregisteredIDs; note that person.centroid_mean has a slight offset 
            # due to the very last element included in the mean calculation
            if deregisteredIDs is not None:
                for person_id in deregisteredIDs:
                    person = tracked_people.get(person_id, None)

                    # if a person is valid (entered from the north of original_frame) and has not paid
                    # check if entering or exiting to update bus state
                    centroids = person.centroids
                    direction = centroids[-1][1] - person.centroid_mean
                    if not person.has_paid and person.is_valid:
                        if direction < 0 and centroids[-1][1] < line_location:
                            self.bus_state.exit_suss()
                            self.bus_state.curr_suss_count -= 1
                            print('[INFO] EXIT SUSS')
                        elif direction > 0 and centroids[-1][1] > line_location:
                            print('[INFO] ENTER SUSS')
                            self.bus_state.enter_suss()

                    if direction < 0 and centroids[-1][1] < line_location:
                        self.bus_state.exit()
                        print('[INFO] EXIT SUSS')
                    # elif direction > 0 and centroids[-1][1] > line_location:
                        # print('[INFO] ENTER SUSS')
                        # bus_state.enter()


            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                person = tracked_people.get(objectID, None)

                # if there is no existing trackable object, create one
                if person is None:
                    person = PeopleTracker.Passenger(objectID, centroid)

                # otherwise, there is a trackable object so we can utilize it
                # to determine direction
                else:
                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')
                    person.update_mean()
                    direction = centroid[1] - person.centroid_mean
                    person.centroids.append(centroid)

                    # check to see if the object has been counted or not
                    if not person.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line AND the valid passenger (ignore re-id problems for now), count the object
                        if direction < 0 and centroid[1] < line_location and person.is_valid is True:
                            print('[INFO] EXIT PASSENGER')
                            self.bus_state.exit()
                            person.counted = True

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count and validate the object
                        elif direction > 0 and centroid[1] > line_location and person.centroids[0][1] < line_location:
                            print('[INFO] ENTER PASSENGER')
                            self.bus_state.enter()
                            # total_down += 1
                            person.counted = True
                            person.is_valid = True
                            self.bus_state.curr_suss_count += 1

                # store the trackable object in our dictionary
                tracked_people[objectID] = person

            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # if the 'p' key was pressed, add a test signal to the payment_id_queue
            # for those with no app running
            if key == ord("p"):
                self.payment_id_queue.put("0")

            # if the 'n' key was pressed, move to the next stop
            if key == ord("n"):
                self.bus_state.next_stop()

            if key == ord(" "):
                while cv2.waitKey(1) & 0xFF != ord(" "):
                    pass

            # construct a tuple of information we will be displaying on the
            # original_frame
            info = [
                ("Current Unpaid Count: ", self.bus_state.curr_suss_count),
                ("Enter Passenger", self.bus_state.entrants),
                ("Exit Passenger ", self.bus_state.exiters),
                #("Enter suss", self.bus_state.suss_entrants),
                #("Exit suss", self.bus_state.suss_exiters),
                # ("Up", total_up),
                # ("Down", total_down),
                #("Status", self.status),
            ]

            # loop over the info tuples and draw them on our original_frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(original_frame, text, (10, frame_height - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # check to see if we should write the original_frame to disk
            if writer is not None:
                # print('[info] writer writing')
                writer.write(original_frame)

            # show the output original_frame
            cv2.imshow("Frame", original_frame)

            # increment the total number of frames processed thus far and
            # then update the FPS counter
            total_frames += 1
            fps.update()

        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # check to see if we need to release the video writer pointer
        if writer is not None:
            print('[info] released file writer')
            writer.release()

        if not self.is_video_file_stream:
            print('[info] stopping live video stream')
            self.video_stream.stop()
        else:
            print('[info] releasing video file pointer')
            self.video_stream.release()

        # close all open windows
        cv2.destroyAllWindows()
        print('[info] all windows closed')

        # send data to server
        self.bus_state.next_stop()
        return

    def object_detection(self, original_frame, processed_frame):
        """  Detects objects within the frame, if it's a person we append a tracker to it

        Args:
            original_frame: the frame of the video
            processed_frame: the color space of the frame

        Returns:
            None
        """
        #print("OBJECT_DETECTION!")
        (frame_height, frame_width) = original_frame.shape[:2]

        # set the status and initialize our new set of object trackers
        self.status = "Detecting"

        #del self.trackers
        #self.trackers = cv2.MultiTracker_create()


        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        # blob = cv2.dnn.blobFromImage(frame, 1.0, None, (102.9891, 115.9465, 122.7717), False, False)
        blob = cv2.dnn.blobFromImage(processed_frame, swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        boxes = []
        confidences = []
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum confidence
            if confidence > self.CONFIDENCE:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if self.CLASSIFICATIONS[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                # create a new object tracker for the bounding box and add it
                # to our multi-object tracker

                #tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                tracker = self.OPENCV_OBJECT_TRACKERS["csrt"]()

                #self.trackers.add(tracker, processed_frame, (start_x, start_y, end_x - start_x, end_y - start_y))
                boxes.append([start_x, start_y, end_x-start_x, end_y-start_y])
                confidences.append(confidence)
                # cv2.rectangle(original_frame, (start_x - 10, start_y - 10), (end_x + 10, end_y + 10), (255, 0, 0), 5)

        boxes = np.array(boxes)
        confidences = np.array(confidences)
        return boxes, confidences

    def deepsort_tracking(self, bboxes, confidences, frame):
        """ Run the deepsort tracking and draw the tracking bounding boxes

        Note: Only one terminal is supported

        Args:
            bboxes: bounding boxes
            confidences: the confidence of each detection
            frame: the image frame

        Returns:
            None
        """
        terminal = self.terminals[0]
        if bboxes.shape[0] != 0:
            ds_tracker, ds_detections_class = self.deepsort.run_deep_sort(frame, confidences, bboxes)
            # print("# of detections : {} and # of trackers : {}".format(len(ds_detections_class), len(ds_tracker.tracks)))
            # print(len(ds_tracker.tracks))

            #print("tracker length =", len(ds_tracker.tracks))
            distance_to_terminal = [None] * len(ds_tracker.tracks)
            tracker_count = 0
            for track in ds_tracker.tracks:
                tracker_count+=1
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr() #Get the corrected/predicted bounding box
                id_num = str(track.track_id) #Get the ID for the particular track.
                features = track.features #Get the feature vector corresponding to the detection.

                centroid = track.get_centroid()
                foot_point = track.get_foot_point()

                # Add to list which checks if this passenger was the one that paid
                if track.has_paid is False and track.is_passenger is True:
                    #print("Set distance")
                    distance_to_terminal[tracker_count-1] = math.sqrt( (terminal[0] - centroid[0])**2 + (terminal[1] - centroid[1])**2 )
                        #math.hypot(terminal[0] - centroid[0], terminal[1] - centroid[1])

                centroid_side = self.check_centroid_side(foot_point)

                # CHECK PASSENGER STATUS######################################
                #not a passenger on not passenger side
                if not track.is_passenger and centroid_side is not self.PASSENGER_SIDE:
                    track.is_passenger = False

                #not a passenger on passenger side (person entered)
                elif not track.is_passenger and centroid_side is self.PASSENGER_SIDE:
                    print("Passenger Entered")
                    track.is_passenger = True
                    self.bus_state.entrants += 1
                    if track.has_paid is False:
                        self.bus_state.curr_suss_count += 1

                #passenger on passenger side
                elif track.is_passenger and centroid_side is self.PASSENGER_SIDE:
                    track.is_passenger = True

                #passenger not on passenger side (person exit)
                elif track.is_passenger and centroid_side is not self.PASSENGER_SIDE:
                    print("Passenger Exit")
                    track.is_passenger = False
                    self.bus_state.exiters += 1

                    if not track.has_paid:
                        self.bus_state.curr_suss_count -= 1
                # END CHECK PASSENGER STATUS #####################################

                # Get color depending on the status of the tracked object
                if track.has_paid:
                    color = (0, 255, 0)  # green
                elif track.is_passenger:
                    color = (0, 0, 255)  # red
                else:
                    color = (100, 255, 255)  # yellow

                # test = str(track.mean.tostring())
                # I think it's direction?
                # output_text = str(id_num) + " " + str(int(round(track.mean[0]))) + " " + str(int(round(track.mean[1]))) + " " + str(int(round(track.mean[2]))) + " " + str(int(round(track.mean[3])))
                output_text = str(id_num)

                # Draw bbox from tracker.
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(frame, output_text, (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)
                cv2.circle(frame, (foot_point[0], foot_point[1]), 4, color, -1)

                # Draw bbox from detector. Just to compare. TYLER maybe keep?
                for det in ds_detections_class:
                    bbox = det.to_tlbr()
                    #cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)

            # check if passenger entered the bus
            # For testing
            #foot_point=ds_tracker.tracks[0].get_foot_point()
            #print(self.check_centroid_side(foot_point))

            try:
                msg = int(self.payment_id_queue.get_nowait())
                print(f"[INFO] PeopleTracker: Payment Signal Received: {msg}")
                self.payment_id_queue.task_done()

                curr_idx = 0
                min_distance = None
                closest_terminal_idx = None

                # iterate through each tracker distance from terminal to get the min distance
                #print("distnace to terminal legnth: ", len(distance_to_terminal))
                for distance in distance_to_terminal:
                    #print("distance: ", distance)
                    if min_distance is None:
                        min_distance = distance
                        closest_terminal_idx = curr_idx
                        print("set min distance to ", curr_idx)
                    if distance is not None and distance < min_distance:
                        min_distance = distance
                        closest_terminal_idx = curr_idx
                        print("set min distance to ", curr_idx)
                    curr_idx += 1

                if distance is not None:
                    ds_tracker.tracks[closest_terminal_idx].has_paid = True
                    self.bus_state.curr_suss_count -= 1
                else:
                    print("failed to find paid customer")
                    self.payment_id_queue.put(msg)
            except queue.Empty:
                pass


    @staticmethod
    def region_of_interest(original_frame, processed_frame, vertices):
        """Applies an image mask based on the ROI on the processed_image

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.

        Args:
            original_frame: original camera frame, used to draw and display back to the user
            processed_frame: the frame to apply masking on
            vertices: array of vertices to draw the polynomial

        Returns:
            cv2.Mat: The image, area outside of the ROI is blackened out
        """

        # Draw the ROI onto the frame
        # https://medium.com/@rndayala/drawing-over-images-9789838ef558
        # reshape points in required format of polylines
        polylines_vertices = vertices.reshape((-1, 1, 2))
        cv2.polylines(original_frame, [polylines_vertices], True, (255, 255, 255), 1)
        #cv2.imshow("Frame", original_frame)  # For testing

        # Format needed for polyfill
        vertices = [vertices]

        # defining a blank mask to start with
        mask = np.zeros_like(processed_frame)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(processed_frame.shape) > 2:
            channel_count = processed_frame.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(processed_frame, mask)
        # cv2.imshow("Frame", masked_image)  # for testing

        return masked_image
