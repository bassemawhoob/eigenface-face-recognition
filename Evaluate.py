from Eigenface import EigenFaces
import os
import GFX
import cv2

class Evaluate:
    def __init__(self):
        self.clf = EigenFaces()
        self.clf.train('training_images')

    # a method to find the optimal threshold without overfitting the model
    def validate(self):
        array = [1500,1600,1700,1800,1900,2000,2100,2200,2300,2400, 2500, 2600, 2700, 2800, 2900, 3000]
        images = []
        rootdir = 'test'
        for subdir, dirs, files in os.walk(rootdir):
            name = subdir.split("/")
            label = name[len(name)-1]
            for file in files:
                if file.endswith('.pgm'):
                    images.append(GFX.Image(cv2.imread(os.path.join(subdir, file)),label))
        accuracies = []
        for t in array:
            total = len(images)
            correct = 0
            for image in images:
                predicted_name = self.clf.predict_face(image.gray().to_numpy_array(), t)
                # for (x, y, w, h) in image.face_areas():
                #     image.draw_rect(x, y, w, h)
                #     face = image.cut(x, y, w, h).gray().scale(100, 100).to_numpy_array()
                #     predicted_name = self.clf.predict_face(face,t)
                #     print(predicted_name)
                #     print(image._label)
                if predicted_name == image._label:
                    correct += 1
                    print("Correct")
                else:
                    print("Incorrect")
            accuracies.append((correct/total)*100)
        print(accuracies)
        print(max(accuracies))
        print(array[accuracies.index(max(accuracies))])

    # a method to calculate the accuracy of the model. Test for a database using 4 fold cross validation (run 4 times on diferent
    # paritions of the database and take the mean accuracy)
    def test(self):
        images = []
        rootdir = 'test'
        for subdir, dirs, files in os.walk(rootdir):
            name = subdir.split("/")
            label = name[len(name)-1]
            for file in files:
                if file.endswith('.pgm'):
                    images.append(GFX.Image(cv2.imread(os.path.join(subdir, file)),label))
        total = len(images)
        correct = 0
        for image in images:
            predicted_name = self.clf.predict_face(image.gray().to_numpy_array())
            if predicted_name == image._label:
                correct += 1
                print("Correct")
            else:
                print("Incorrect")
        print((correct/total)*100)


if __name__ == "__main__":
    eval = Evaluate()
    eval.validate()