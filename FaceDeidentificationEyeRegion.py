from FacialLandmarkDetection import *
from Database_loader import *

#method shows all database images in windows

def showAllDatabaseImages(database_folder, extension="ppm", imageNum=""):
    loader = DatabaseLoaderXMVTS2(database_folder)
    images_paths = loader.loadDatabase(extension, imageNum)
    print(images_paths)
    for imagePath in images_paths:
        detector = FacialLandmarkDetector(imagePath)
        detector.showImage()
def saveImage(imagePath, destination):
    imgName = imagePath.split("/")[-1]
    img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    cv2.imwrite(destination+"/"+imgName,img)

def storeDatabaseImagesToDestination(database_folder, destination, extension="ppm", ImageNum=""):
    loader = DatabaseLoaderXMVTS2(database_folder)
    images_paths = loader.loadDatabase(extension, ImageNum)
    for imagePath in images_paths:
        saveImage(imagePath, destination)
    print("Saving images finished!")
def findFacialLandmarksOnTemplateImages(templates_folder, destination, showImages = False, store=False, storePositions = False):
    loader = DatabaseLoaderXMVTS2(templates_folder)
    images_paths = loader.loadDatabase("ppm", "")
    landmarksPositions = []
    for imagePath in images_paths:
        detector = FacialLandmarkDetector(imagePath)
        positions = detector.detectFacialLandmarks(True)
        landmarksPositions.append(positions)
        if store == True:
            detector.saveImage(destination)
        if storePositions == True:
            text = ' '.join('%s %s' % x for x in positions)
            imageName = imagePath.split("/")[-1].split(".")[0]
            dirName = "/".join(imagePath.split("/")[:-1])
            f=open(dirName + "/" + imageName+".txt",'w')
            f.write(text)
            print("Stored result for image " + imageName)
            f.close()
        if showImages==True:
            detector.showImage()
    print("Finished finding facial landmarks on templates.")
def findFacialLandmarksOnTemplateImages_EyeRegion(templates_folder, destination, showImages = False, store=False, storePositions = False):
    loader = DatabaseLoaderXMVTS2(templates_folder)
    images_paths = loader.loadDatabase("ppm", "")
    landmarksPositions = []
    for imagePath in images_paths:
        detector = FacialLandmarkDetector(imagePath)
        ROI = detector.extractFacePart("EyeRegion")
        #landmarksPositions.append(positions)
        #if store == True:
         #   detector.saveImage(destination)
        #if storePositions == True:
         #   text = ' '.join('%s %s' % x for x in positions)
          #  imageName = imagePath.split("/")[-1].split(".")[0]
          #  dirName = "/".join(imagePath.split("/")[:-1])
            #f=open(dirName + "/" + imageName+".txt",'w')
            #f.write(text)
            #print("Stored result for image " + imageName)
            #f.close()
        if showImages==True:
            cv2.imshow('image',ROI)
            cv2.waitKey(0)
    print("Finished finding facial landmarks on templates.")
def getTemplatePaths(templates_folder):
    fileNames = []
    for root, dirs, files in os.walk(templates_folder):
        for file in files:
            if file.endswith(".txt"):
                fName = os.path.join(root, file)
                fileNames.append(fName)
    fileNames = sorted(fileNames)
    return fileNames
def loadTemplatesPositions(templates_folder):
    positions = []
    fileNames = getTemplatePaths(templates_folder)
    for fileName in fileNames:
        position = []
        f = open(fileName,"r")
        line = f.readline()
        tempPos = line.split(" ")
        for i in range(0, len(tempPos), 2):
            position.append((float(tempPos[i]), float(tempPos[i+1])))
        positions.append(position)
        f.close()

    return positions
def find_closest_Image(image_positions, templatePositions):
    closest = -1
    minScore = 1111111111111111
    N = len(image_positions)
    i=0
    for template in templatePositions:
        difference = sum(sum(numpy.subtract(image_positions, template)**2)/N)
        if difference < minScore:
            minScore = difference
            closest = i
        i += 1
    print (minScore)
    return closest
def loadDatabaseImage_CalculateFacialLandmarks(database_folder, imageName, showImages=False):
    loader = DatabaseLoaderXMVTS2(database_folder)
    imagePath = loader.imagePathFinder(imageName, "")
    detector = FacialLandmarkDetector(imagePath)
    positions = detector.detectFacialLandmarks(True)
    if showImages==True:
        detector.showImage()
    return positions

if __name__ == "__main__":
    XMVTS2_database = "/home/matej/FER_current/Projekt/Project_Deidentification_Kazemi/baza_XMVTS2"
    destination = "/home/matej/FER_current/Projekt/Project_Deidentification_Kazemi/baza_deidentification_Images"
    templates_database = "/home/matej/FER_current/Projekt/Project_Deidentification_Kazemi/baza_templates"
    templates_destination = "/home/matej/FER_current/Projekt/Project_Deidentification_Kazemi/KazemiTemplates"

    #findFacialLandmarksOnTemplateImages(templates_database, templates_destination, False, False, True)
    #findFacialLandmarksOnTemplateImages_EyeRegion(templates_database, templates_destination, True, False, False)
    templates_positions = loadTemplatesPositions(templates_database)
    image_positions = loadDatabaseImage_CalculateFacialLandmarks(destination, "001", True)
    closest_Index = find_closest_Image(image_positions, templates_positions)
    closest_Image_path = getTemplatePaths(templates_database)[closest_Index]
    print(closest_Image_path)
