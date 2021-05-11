# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:

    def __init__(self, maxdisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxdisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectid):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectid]
        del self.disappeared[objectid]

    def update(self, rects):
        if len(rects) == 0:  # check to see if the list of input bounding box rectangles is empty
            for objectid in list(self.disappeared.keys()): # loop over any existing tracked objects and mark disappeared
                self.disappeared[objectid] += 1
                # if reached maximum number of consecutive frames where object has been marked as missing, deregister
                if self.disappeared[objectid] > self.maxDisappeared:
                    self.deregister(objectid)
                    return self.objects  # return early as there are no centroids or tracking info to update

        inputcentroids = np.zeros((len(rects), 2), dtype="int")  # initialize array of input centroids for current frame
        for (i, (startX, startY, endX, endY)) in enumerate(rects):  # loop over the bounding box rectangles
            cx = int((startX + endX) / 2.0)
            cy = int((startY + endY) / 2.0)
            inputcentroids[i] = (cx, cy)  # use the bounding box coordinates to derive the centroid

        # if we are currently not tracking any objects take the inpu centroids and register each of them
        if len(self.objects) == 0:
            for j in range(0, len(inputcentroids)):
                self.register(inputcentroids[j])
        else:  # try to match existing objects to input
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute the distance between each pair of object centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing object centroid
            d = dist.cdist(np.array(object_centroids), inputcentroids)

            # in order to perform this matching we must (1) find the smallest value in each row and then (2)
            # sort the row indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index list
            rows = d.min(axis=1).argsort()

            # next, we perform a similar process on the columns by finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = d.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            used_rows = set()
            used_cols = set()

            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or column value before, ignore it val
                if row in used_rows or col in used_cols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared counter
                objectid = object_ids[row]
                self.objects[objectid] = inputcentroids[col]
                self.disappeared[objectid] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                used_rows.add(row)
                used_cols.add(col)

                # compute both the row and column index we have NOT yet examind
                unused_rows = set(range(0, D.shape[0])).difference(used_rows)
                unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:  # loop over the unused row indexes
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectid = object_ids[row]
                    self.disappeared[objectid] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectid] > self.maxDisappeared:
                        self.deregister(objectid)

                    # otherwise, if the number of input centroids is greater
                    # than the number of existing object centroids we need to
                    # register each new input centroid as a trackable object
                    else:
                        for col in unused_cols:
                            self.register(inputcentroids[col])

        # return the set of trackable objects
        return self.objects
