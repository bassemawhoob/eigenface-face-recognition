'''
Implementation of Eigenfaces face recognition using PCA.

All images must be the same size and converted to grayscale.
Each person/class must have his/her own folder, and every photo of a given
person must be placed into the same folder, and the folder must be given the
person/class's name.

The algorithm is run from the main.py module.
When run, the algorithm will print the predicted classification to the console
and will show the picture.
'''

from PIL import Image
import numpy as np
import sys
import os

import PCA

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2

class EigenFaces(object):
    def train(self, root_training_images_folder):
        self.projected_classes = []

        self.list_of_arrays_of_images, self.labels_list, \
            list_of_matrices_of_flattened_class_samples = \
                read_images(root_training_images_folder)

         # create matrix to store all flattened images
        images_matrix = np.array([np.array(Image.fromarray(img)).flatten() for img in self.list_of_arrays_of_images],'f')

        # perform PCA
        self.eigenfaces_matrix, variance, self.mean_image = PCA.pca(images_matrix)

        # Projecting each class sample (as class matrix) and then using the class average as the class weights for comparison with the Target image
        for class_sample in list_of_matrices_of_flattened_class_samples:
            class_weights_vertex = self.project_image(class_sample)
            self.projected_classes.append(class_weights_vertex.mean(0))

    def project_image(self, X):
        X = X - self.mean_image
        return np.dot(X, self.eigenfaces_matrix.T)

    def predict_face(self, X,t = 2700):
        min_class = -1
        min_distance = np.finfo('float').max
        projected_target = self.project_image(X)
        # delete last array item, it's nan
        projected_target = np.delete(projected_target, -1)
        for i in range(len(self.projected_classes)):
            distance = np.linalg.norm(projected_target - np.delete(self.projected_classes[i], -1))
            if distance < min_distance:
                min_distance = distance
                min_class = self.labels_list[i]
        if min_distance < t:
            return min_class
        else:
            return "Unseen"


    def predict_faces(self, image_path):
        names = []
        # Returns all the faces in a picture aligned for prediction
        aligned_faces = face_align(image_path)
        for face in aligned_faces:
            image = np.array(Image.fromarray(face).convert('L'), dtype=np.uint8).flatten()
            names.append(self.predict_face(image))
        return names

def read_images(path, sz=None):
    class_samples_list = []
    class_matrices_list = []
    images, image_labels = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            class_samples_list = []
            for filename in os.listdir(subject_path):
                if filename != ".DS_Store":
                    try:
                        im = Image.open(os.path.join(subject_path, filename))
                        if (sz is not None):
                            im = im.resize(sz, Image.ANTIALIAS)
                        images.append(np.asarray(im, dtype = np.uint8))

                    except IOError as e:
                        errno, strerror = e.args
                        print("I/O error({0}): {1}".format(errno, strerror))
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        raise
                    # adds each sample within a class to this List
                    class_samples_list.append(np.asarray(im, dtype = np.uint8))

            # flattens each sample within a class and adds the array/vector to a class matrix
            class_samples_matrix = np.array([img.flatten() for img in class_samples_list], 'f')

             # adds each class matrix to this MASTER List
            class_matrices_list.append(class_samples_matrix)

            image_labels.append(subdirname)

    return images, image_labels, class_matrices_list

def face_align(img_path):

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=100)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img_path)
    # image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    # cv2.imshow("Input", image)
    rects = detector(gray, 2)
    images =[]

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceOrig = image[y:y + h, x:x + w]
        faceAligned = fa.align(image, gray, rect)
        images.append(faceAligned)

        # import uuid
        # f = str(uuid.uuid4())
        # cv2.imwrite("foo/" + f + ".png", faceAligned)
        #
        # display the output images
        # cv2.imshow("Original", faceOrig)
        # cv2.imshow("Aligned", faceAligned)
        # cv2.waitKey(0)

    return images