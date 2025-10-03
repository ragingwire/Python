import os
import sys
from PIL import Image



class File ( object ):
    
    def __init__ ( self, fullPath  ):
        super ().__init__ ()
        self._fullPath = fullPath
        self._rootPath = ""
        self._name = ""
        self._extension = ""
        self._exists = False
        self.__splitName__ ()
        self.exists ()


    def getName ( self ):
        return self._name 
    
    def setFullPath ( self, fullPath ):
        self._fullPath = fullPath
        
    def getFullPath ( self ) :
        return self._fullPath
    
    def getRootPath ( self ):
        return self._rootPath
    
    def setExtension (self, extension ):
        self._extension = extension
        self._fullPath = self._rootPath + extension
    
    def getExtension ( self ):
        return self._extension
    
    def exists ( self ):
        self._exists = os.path.exists( self._fullPath )
        return self._exists
    
    def create (self, attribute = "x" ):
        try :
            with open( self._fullPath, "x") as f:
                self._exists = True
        except FileNotFoundError :
            self._exists = False
        except FileExistsError :
            self._exists = True
        finally:
            self._exists = False
        
        return self._exists
    
    def __splitName__ ( self ):
        self._name = os.path.basename( self._fullPath )
        self._rootPath, self._extension = os.path.splitext ( self._fullPath )
                
 
class ImageFile ( File ):
     
     _IMAGE_TYPE_UNDEFINED = "undefined"
     _IMAGE_TYPE_PNG = ".png"
     _IMAGE_TYPE_JPG = ".jpg"
     _IMAGE_TYPE_JPEG = ".jpeg"
     _IMAGE_TYPE_GIF = ".gif"
     
     def __init__ ( self, fullPath ):
         super ().__init__ ( fullPath )
         self._quality = 100
         self._imageType = self._IMAGE_TYPE_UNDEFINED
         self.__determineImageType__ ()
         
     def isPNG ( self ):
        return self._imageType == self._IMAGE_TYPE_PNG   
     
     def setJPG (self, quality = 95 ):
         self._fullPath = self._rootPath + self._IMAGE_TYPE_JPG
         self.__splitName__ ()
         self._imageType = self._IMAGE_TYPE_JPG
         self.exists ()
         
     def isJPG ( self ):
        return self._imageTpe == self.__IMAGE_TYPE_JPG
    
     def  isGIF ( self ):
        return self._imageTpe == self.__IMAGE_TYPE_GIF
    
     def getQuality (self ) :
         return self._quality
     
     def __determineImageType__ ( self ):
        lowercase_extension = self._extension.lower ()
        if lowercase_extension ==self._IMAGE_TYPE_PNG :
            self._imageType = self._IMAGE_TYPE_PNG
        elif lowercase_extension == self._IMAGE_TYPE_JPG or lowercase_extension == self._IMAGE_TYPE_JPEG :
            self._imageType = self._IMAGE_TYPE_JPG
        elif lowercase_extension == self._IMAGE_TYPE_GIF :
            self._imageType = self._IMAGE_TYPE_GIF
        
               

class Directory ( File ):
    def __init__ (self, fullPath ):
        super ().__init__ ( fullPath )
        self._fileList = []
        self._fullPathFileList = []
        self._numFiles = 0
        
    def getName (self ):
        return self._fullPath
    
    def setName (self, fullPath ):
        super ().setFullPath ( fullPath )
        self.getFiles ()
        return self.exists ()
    
    def exists ( self ) :
        if not os.path.exists( self._fullPath ):
            self._exists = False
        return self._exists
    
    def create (self ):
        return self.exists
    
    def __getFiles__ ( self ):
        try :
            self._fileList = os.listdir( self._fullPath  )
            self._numFiles = len ( self._fileList )
        except FileNotFoundError :
            self._exists = False;
            self._numFiles = 0;
            self._fileList = []
        return self._fileList
    
    def getFiles ( self ) :
        self.__getFiles__ ()
        for i in range (self._numFiles ):
            full_path_name = self._fullPath +"\\" + self._fileList [ i ]
            self._fullPathFileList.append ( full_path_name )
        return self._fullPathFileList
    
    def getNumFiles ( self ):
        return self._numFiles

class ImageProcessor ( object ):
    
    def __init__ (self, application = None  ) :
        super ().__init__ ()
        self._imageProcessed = 0
        self._imageConverted = 0
        self._imageSkipped = 0
        self._imageError = 0
        self._imageScaled = 0
        self._application = application
        
    def getNumProcessedImages ( self ):
        return self._imageProcessed
    
    def getNumConvertedImages ( self ):
        return self._imageConverted
    
    def getNumSkippedImages ( self ):
        return self._imageSkipped
    
    def getNumErrorImages ( self ) :
        return self._imageError
    
    def getNumScaledImages ( self ) :
        return self._imageScaled
    
    def convertPNGtoJPG ( self, sourceImage, destinationImage, quality = 95 ):
        if sourceImage.exists () and sourceImage.isPNG () :
            try:
                source = sourceImage.getFullPath () 
                destination = destinationImage.getFullPath ()
                self.__log__ ( "processing file: " + source )
                with Image.open( sourceImage.getFullPath () ) as Img:
                    if Img.mode == 'RGBA':
                        Img = Img.convert('RGB')
                    #Img.show ()
                    Img.save ( destinationImage.getFullPath (), 'jpeg', quality = destinationImage.getQuality () )
                    self._imageConverted += 1
            except Exception as e:
                self._imageSkipped += 1
                self._imageError += 1
                return False
        else :
            self._imageSkipped +=1
            self.__log__ ( "skipping non PNG file " + sourceImage.getFullPath () )
            return False
            
        return True
    
    def __log__ ( self, logstr ):
        self._application.log ( logstr )

class Application ( object ):
    
    def __init__ (self, applicationName ):
        super ().__init__ ()
        self.applicationName = applicationName
        self.commandLineArgs = []
        self.options = []
        self.numCommandLineArgs = 0
        
    def getApplicationName ( self ) :
        return self.ApplicationName
    
    def getNumCommandLineArgs (self):
        return self.numCommandLineArgs
    
    def printUsage ( self ):
        pass
    
    def log (self, logstr ):
        print ( logstr )
    
    def __run__ (self ):
        pass
    
    def exit (self,exitCode = 0 ):
        self.exitCode = exitCode
        sys.exit ( self.exitCode )
        
    def __handeCommandLineArgs__ ( self ):
        pass
    

class ImageProcessorApplication ( Application ):
    
    APPLICATION_NAME = "Image Processor"
    APPLICATION_USAGE = "Usage: python your_script_name.py <input_directory_path>"
    
    def __init__ (self ):
        super ().__init__ ( ImageProcessorApplication.APPLICATION_NAME )
        self.sourceDirectory = Directory ( "" )
        self.destinationDirectory = Directory ( "" )
        self.__handleCommanddLineArgs__()
        self.__run__ ()
    
    def __handleCommanddLineArgs__ ( self ):
        self.numCommandLineArgs = len ( sys.argv )
        if self.numCommandLineArgs == 1:
            self.__printUsage__ ()
            return False
        elif self.numCommandLineArgs == 2:
            self.sourceDirectory.setName ( sys.argv [ 1 ] )
            self.destinationDirectory.setName ( sys.argv [ 1 ] )
        elif self.numCommandLineArgs == 3:
            self.sourceDirectory.setName ( sys.argv [ 1 ] )
            self.destinationDirectory.setName ( sys.argv [ 2 ] )
        
        return True
    
    def __printUsage__ (self ):
        return True
    
    def __run__ ( self ):
        imageProcessor = ImageProcessor ( self )
        imageFiles = self.sourceDirectory.getFiles ()
        images_to_process = self.sourceDirectory.getNumFiles ()
        for i in range( images_to_process ) :
            imageName = imageFiles [ i ]
            sourceImage = ImageFile ( imageName )
            destinationImage = ImageFile ( imageName )
            destinationImage.setJPG ( quality = 95 )
            imageProcessor.convertPNGtoJPG ( sourceImage, destinationImage )
        
            
        print ( "Images processed: " + str (imageProcessor.getNumProcessedImages () ) ) 
        print ( "Images converted " + str (imageProcessor.getNumConvertedImages () ) )
        print ( "Images skipped: " + str (imageProcessor.getNumSkippedImages () ) )  
        print ( "Images with errors: " + str (imageProcessor.getNumErrorImages () ) )    
        return True
    


if __name__ == "__main__":
    
    # aba ja 
    imageProcessorApplication = ImageProcessorApplication ()
    
   