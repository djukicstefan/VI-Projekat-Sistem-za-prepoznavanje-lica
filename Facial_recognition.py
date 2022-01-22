# ukljucivanje odgovarajucih biblioteka
from turtle import left, right
import face_recognition
import argparse
import pickle
import cv2

# konstruisanje parsera argumenata
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# ucitavanje vec poznatih lica i njihovih mera iz baze podataka
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# ucitavanje slike sa diska pomocu OpenCV biblioteke i promena redosleda boja
# iz BGR formata u RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detektovanje (x, y) koordinata okvira
# koji odgovaraju svakom licu ponaosob na slici
# odredjivanje mera lica primenom istrenirane neuronske mreze za svako lice    
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

#inicijalizacija liste koja ce sadrzati imena svih prepoznatih lica
names = []

# petlja koja prolazi kroz mere lica
for encoding in encodings:
    # uporedjivanje da li prepoznata lica sa slike
    # na ulazu imaju poklapanja u bazi podataka
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    # provera da li ima poklapanja
    if True in matches:
        # deo koji prolazi sve indekse prepoznatih lica
        # i koristi dictionary kao strukturu podataka za 
        # odredjivanje koliko puta je svako lice prepoznato
        matchedIdxs = [i for(i, b) in enumerate(matches) if b]
        counts = {}

        # petlja koja prolazi kroz indekse koji predstavljalju prepoznata lica
        # i izracunava ukupan broj prepoznavanja za svako lice
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        
        # odredjivanje o kom prepoznatom licu se radi
        # na osnovu broja glasova primenom k-NN klasifikacije
        name = max(counts, key=counts.get)

    # azuriranje liste imena
    names.append(name)

# petlja koja prolazi kroz prepoznata lica
for ((top, right, bottom, left), name) in zip(boxes, names):
    # upisivanje prepoznatog imena
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# prikaz kranjnje slike i rezultata
cv2.imshow("Image", image)
cv2.waitKey(0)