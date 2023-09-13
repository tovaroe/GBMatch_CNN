import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane

int z = 0
int t = 0
def plane = ImagePlane.getPlane(z, t)
def roi = ROIs.createRectangleROI(0, 0, 512, 512, plane)
def annotation = PathObjects.createAnnotationObject(roi)
addObject(annotation)
selectAnnotations();

setImageType('BRIGHTFIELD_H_E');
setColorDeconvolutionStains('{"Name" : "H&E estimated", "Stain 1" : "Hematoxylin", "Values 1" : "0.66577 0.67961 0.30803", "Stain 2" : "Eosin", "Values 2" : "0.24246 0.9302 0.27559", "Background" : " 196 113 189"}');
setPixelSizeMicrons(0.912000, 0.912000)
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImageBrightfield": "Hematoxylin OD",  "requestedPixelSizeMicrons": 0.0,  "backgroundRadiusMicrons": 15.0,  "medianRadiusMicrons": 4.0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 20.0,  "maxAreaMicrons": 1000.0,  "threshold": 0.1,  "maxBackground": 2.0,  "watershedPostProcess": true,  "cellExpansionMicrons": 0.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');


def imageData = getCurrentImageData()

def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
saveDetectionMeasurements('/HE_results/' + name + '.txt')