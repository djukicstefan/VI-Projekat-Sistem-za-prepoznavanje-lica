# ukljucivanje odgovarajucih biblioteka
from imutils import paths
import face_recognition
import pickle
import argparse
import cv2
import os

# konstruisanje parsera argumenata
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input direcotry of faces + images")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# uzimanje putanje do datoteke koja sadrzi skup podataka (slika)
print("[INFO] quiantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# inicijalizacija liste poznatih enkodiranih slika i poznatih imena
knownEncodings = []
knownNames = []

# petlja koja prolazi kroz sve slike
for (i, imagePath) in enumerate(imagePaths):
    # ovde se uzima ime osove iz imena datoteke (potrebno je staviti da ime datoteke bude isto kao i ime osobe)
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # ucitavanje slike sa diska pomocu OpenCV biblioteke i promena redosleda boja
    # iz BGR formata u RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detektovanje (x, y) koordinata okvira
    # koji odgovaraju svakom licu ponaosob na slici    
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

    # odredjivanje mera lica primenom istrenirane neuronske mreze za svako lice
    encodings = face_recognition.face_encodings(rgb, boxes)

    # petlja koja prolazi kroz enkodirane fotografije
    for encoding in encodings:
        # dodavanje svih enkodiranih formata u odgovarajucu listu
        # zajedno sa imenom osobe
        knownEncodings.append(encoding)
        knownNames.append(name)


outDict = {
    "encodings" : knownEncodings,
    "names" : knownNames
}
outfile = open(args["encodings"],'wb')
pickle.dump(outDict, outfile)
outfile.close()        