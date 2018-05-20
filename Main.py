from Eigenface import EigenFaces
from Eigenface import face_align
import numpy as np
from PIL import Image
import datetime
import sys
import os
import cv2
import Webcam
from shutil import copyfile
import GFX

if __name__ == "__main__":

    clf = EigenFaces()
    clf.train('training_images')
    # print(clf.predict_face(np.array(Image.open('test/s1/10.pgm').convert('L'), dtype=np.uint8).flatten()))
    print(clf.predict_face(np.array(Image.open('validate/2.pgm').convert('L'), dtype=np.uint8).flatten()))

    # # # try/ testfull.jpg'
    # image = GFX.Image(cv2.imread('try/testfull.jpg'))
    # face_detector = GFX.FaceDetector()
    # face_detector.show(image, wait=False)


    def ensure_dir_exists(path):
        if not os.path.isdir(path):
            os.mkdir(path)


    def take_training_photos(name, n):
        for i in range(n):
            for face in Webcam.capture().faces():
                normalized = face.gray().scale(100, 100)
                face_path = 'training_images/{}'.format(name)
                ensure_dir_exists(face_path)
                normalized.save_to('{}/{}.pgm'.format(face_path, i + 1))
                # normalized.show()


    def parse_command():
        args = sys.argv[1:]
        return args[0] if args else None


    def print_help():
        print("""Usage:
        train - takes 10 pictures from webcam to train software to recognize your
                face.
        demo - runs live demo. Captures images from webcam and tries to recognize
               faces.
        """)


    def train():
        name = input('Enter your name: ')
        take_training_photos(name, 10)


    def main():
        cmd = parse_command()
        if cmd == 'train':
            train()
        elif cmd == 'demo':
            Webcam.display()
        else:
            print_help()

    def organize():
        rootdir = 'training_images_final'
        for subdir, dirs, files in os.walk(rootdir):
            path = '{}/{}'.format('training_images',subdir)
            if not os.path.isdir(path):
                os.mkdir(path)
            for file in files:
                to_str = '{}/{}/{}'.format('training_images', subdir, file)
                if file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.pgm'):
                    from_str = subdir+'/'+file
                    faces = GFX.Image(cv2.imread(from_str)).faces()
                    for face in faces:
                        normalized = face.gray().scale(100, 100)
                        normalized.save_to(to_str)
                        #copyfile(from_str,to_str)
    def align_all():
        rootdir = 'modifiedfacedb'
        for subdir, dirs, files in os.walk(rootdir):
            path = '{}/{}'.format('alignedfacedb',subdir)
            if not os.path.isdir(path):
                os.mkdir(path)
            for file in files:
                to_str = '{}/{}/{}'.format('alignedfacedb', subdir, file)
                if file.endswith('.jpeg'):
                    from_str = subdir+'/'+file
                    faces = face_align(from_str)
                    for face in faces:
                        Image.fromarray(face).save(to_str)


    # print(faces)

    # train()
    # main()
    # Webcam.display()
    # organize()
    # align_all()