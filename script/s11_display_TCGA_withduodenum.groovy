guiscript=true
// importations
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import java.io.BufferedReader;
import java.io.FileReader;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.roi.RectangleROI;
import qupath.lib.roi.*
import qupath.lib.objects.*
import qupath.lib.gui.measure.ObservableMeasurementTableData
import qupath.lib.geom.Point2
import qupath.lib.gui.QuPathGUI.*
import net.imagej.ImageJ.*
import qupath.lib.gui.viewer.overlays.*
import qupath.lib.gui.images.servers.*
import java.awt.image.BufferedImage;

// get selected image and server
def imageData = getCurrentImageData()
def server = getCurrentServer()


// get the image name and cut it into short name 
def name =  GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
shortname = name[0..14]
print shortname


// load predicted classes file for slide in question
def file2 = "/work/shared/ptbc/CNN_Pancreas_V2/Donnees/EfficientNet_1/Resultats/LamesCompletes/CombModels_BSC_prediction/"+shortname+".csv"
// define path as a file and check if it's available for importation 
check_file = new File(file2).isFile()
if (check_file == false){Dialogs.showPlainMessage('Warning', "Missing file cannot be imported")}

// use FileReader to be able to read the file
def csvReader2 = new BufferedReader(new FileReader(file2))

row = csvReader2.readLine() // first row (header)

// Loop through all the rows of the CSV file.
def tmp_pl = []
def celllist = []
def cellclass = []
while ((row2 = csvReader2.readLine()) != null) {
    def rowContent = row2.split(",")
    celltemp = rowContent[0]; // first col {tile name}
    pathtemp = rowContent[1]; // 2nd col {classes}
    celllist << celltemp
    cellclass << pathtemp
    tmp_pl << celltemp+','+pathtemp
    }

//definir les class//
def classe1 = getPathClass("Normal_0.80")
def classe2 = getPathClass("Normal_0.50")
def classe3 = getPathClass("Stroma_0.80")
def classe4 = getPathClass("Stroma_0.50")
def classe5 = getPathClass("Tumeur_0.80")
def classe6 = getPathClass("Tumeur_0.50")
def classe7 = getPathClass("Duodenum_0.80")
def classe8 = getPathClass("Duodenum_0.50")
def ND = getPathClass('Not_determined')

//QuPathGUI.getInstance().openImage("/work/shared/ptbc/CNN_Pancreas_V2/Donnees/project_kernel03_scan21000500/scan_tiles/"+shortname+"/"+shortname+".qpdata", false, false)
//qupath.lib.io.PathIO.readImageData(new File("/work/shared/ptbc/CNN_Pancreas_V2/Donnees/project_kernel03_scan21000500/scan_tiles/"+shortname+"/"+shortname+".qpdata"), getCurrentImageData(), getCurrentServer(), BufferedImage.class) 


n=1
// go throught qpdata
for (annotation in getDetectionObjects()) 
{
    name = annotation.getName()
    n = name.replaceAll('Tile ','')
    def nametile = shortname+'_'+n // first tile
    annotation.setName(nametile) // git to the annotation the name of tile
    index = celllist.indexOf(nametile)  // get the classe of the given tile  
    name_cell = celllist[index]
    name_class = cellclass[index]

    // color the given tile (according to his class)
    if (name_class == "classe1"){annotation.setPathClass(classe1)}
    if (name_class == "classe2"){annotation.setPathClass(classe2)}
    if (name_class == "classe3"){annotation.setPathClass(classe3)}
    if (name_class == "classe4"){annotation.setPathClass(classe4)}
    if (name_class == "classe5"){annotation.setPathClass(classe5)}
    if (name_class == "classe6"){annotation.setPathClass(classe6)}
    if (name_class == "classe7"){annotation.setPathClass(classe7)}
    if (name_class == "classe8"){annotation.setPathClass(classe8)}
    if (name_class == "NA"){annotation.setPathClass(ND)}
    n = n+1
}

// Normal (80/50)
classe1.setColor(getColorRGB(20, 148, 20)) // green
classe2.setColor(getColorRGB(165, 209, 82)) // clear green
//Stroma (80/50)
classe3.setColor(getColorRGB(1, 49, 180)) //blue
classe4.setColor(getColorRGB(49, 140, 231)) // clear blue
// Tumor (80/50)
classe5.setColor(getColorRGB(255,51,0)) // red
classe6.setColor(getColorRGB(254,152,113)) // clear red
// Duodenum (80/50)
classe7.setColor(getColorRGB(254, 246, 0)) // yellow
classe8.setColor(getColorRGB(255, 250, 90)) // clear yellow


ND.setColor(getColorRGB(86,86,86)) // gray

print ('Done !')
















