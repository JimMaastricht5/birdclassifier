# code is from PyimageSearch object tracking lesson.  see link below for more details
# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
# code was reformatted to fit python programing standards and for my own understanding
# Jim Maastricht 2021
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:

    # initialize the next unique object ID along with two ordered dictionaries used to keep track objects
    # store the number of maximum consecutive frames a given object is allowed to be marked as "disappeared" until
    # the object is deregistered from tracking
    def __init__(self, maxdisappeared=10):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.objnames = OrderedDict()
        self.objconfidences = OrderedDict()
        self.rects = OrderedDict()
        self.maxDisappeared = maxdisappeared

    # register an object with next available object ID
    def register(self, centroid, rect, objconfidence, objname):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.objconfidences[self.nextObjectID] = objconfidence
        self.objnames[self.nextObjectID] = objname
        self.rects[self.nextObjectID] = rect
        self.nextObjectID += 1

    # deregister an object ID by deleting the ID from both dictionaries
    def deregister(self, objectid):
        del self.objects[objectid]
        del self.disappeared[objectid]
        del self.objconfidences[objectid]
        del self.objnames[objectid]
        del self.rects[objectid]

    # update the object dictionaries with the newly detected objects and rectangles
    # expects object of type list as input
    def update(self, rects, objconfidences, objnames):
        if len(rects) == 0:  # check to see if the list of input bounding box rectangles is empty
            for objectid in list(self.disappeared.keys()):  # loop over existing tracked objects and mark disappeared
                self.disappeared[objectid] += 1
                if self.disappeared[objectid] > self.maxDisappeared:  # object is missing from max consecutive frames
                    self.deregister(objectid)
            return

        inputcentroids = np.zeros((len(rects), 2), dtype="int")  # initialize array of input centroids for current frame
        for (i, (startX, startY, endX, endY)) in enumerate(rects):  # loop over the bounding box rectangles
            cx = int((startX + endX) / 2.0)
            cy = int((startY + endY) / 2.0)
            inputcentroids[i] = (cx, cy)  # use the bounding box coordinates to derive the centroid

        # if we are currently not tracking any objects take the input centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputcentroids)):
                self.register(inputcentroids[i], rects[i], objconfidences[i], objnames[i])
        else:  # try to match existing objects to input
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute the distance between each pair of object centroids and input centroids, respectively
            # (1) find the smallest value in each row and then (2) sort the row indexes based on their minimum values
            # so that the row with the smallest value is at the *front* of the index list (3) performsame on columns
            d = dist.cdist(np.array(object_centroids), inputcentroids)
            rows = d.min(axis=1).argsort()
            cols = d.argmin(axis=1)[rows]

            # in order to determine if we need to update, register, or deregister an object we need to
            # keep track of which of the rows and column indexes we have already examined
            used_rows = set()
            used_cols = set()

            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or column value before, ignore it
                if row in used_rows or col in used_cols:
                    continue

                # otherwise, grab the object ID for the current row, set its new centroid, reset disappeared counter
                objectid = object_ids[row]
                self.objects[objectid] = inputcentroids[col]
                self.disappeared[objectid] = 0
                # indicate that we have examined each of the row and column indexes, respectively
                used_rows.add(row)
                used_cols.add(col)

            # check to see if some of these objects have potentially disappeared
            # compute both the row and column index we have NOT yet examined
            unused_rows = set(range(0, d.shape[0])).difference(used_rows)
            unused_cols = set(range(0, d.shape[1])).difference(used_cols)
            if d.shape[0] >= d.shape[1]:
                for row in unused_rows:  # loop over the unused row indexes
                    # grab the object ID for the corresponding row index and increment the disappeared counter
                    objectid = object_ids[row]
                    self.disappeared[objectid] += 1
                    # check to number of consecutive frames the object has been marked "disappeared"
                    if self.disappeared[objectid] > self.maxDisappeared:
                        self.deregister(objectid)
            else:  # otherwise input > existing centroid, register new trackable object
                for col in unused_cols:
                    self.register(inputcentroids[col], rects[col], objconfidences[col], objnames[col])
        return


def main():
    obj_tracker = CentroidTracker()
    rects = list([])
    rects.append((5, 15, 5, 15))
    rects.append((80, 80, 180, 180))
    print(type(rects))
    print(rects)
    objconfidences, objnames = [], []
    objconfidences.append(120)
    objconfidences.append(200)
    objnames.append('bird 1')
    objnames.append('bird 2')
    obj_tracker.update(rects, objconfidences, objnames)
    print(obj_tracker.objects, obj_tracker.rects, obj_tracker.objnames, obj_tracker.objconfidences)

    for i in (10, 20, 30, 40, 50):
        rects = []
        rects.append((0 + i, 10 + i, 0 + i, 10 + i))
        rects.append((100 - i, 100 - i, 200 - i, 200 - i))
        obj_tracker.update(rects, objconfidences, objnames)
        print(obj_tracker.objects, obj_tracker.disappeared)

    for i in range(1, 60):
        rects = []
        print(i)
        # obj_tracker.update(rects, objconfidences, objnames)
        obj_tracker.update([], [], [])
        print(obj_tracker.objects)

    print('for loop')
    print(obj_tracker.rects)
    for key in obj_tracker.rects:
        print(key, obj_tracker.rects[key])


if __name__ == "__main__":
    main()
