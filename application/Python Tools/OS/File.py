import os



class File ( object ):
    
    def __init__ ( self, fullPath  ):
        super ().__init__ ()
        self._fullPath = fullPath
        self._rootPath = ""
        self._name =""
        self._extension = ""
        self._exists = False
        self.__splitName__ ()

    def setName ( self, name ):
        self._name = name
        
    def getName ( self ):
        return self._name 
    
    def setFullPath ( self, fullPath ):
        self._fullPath = fullPath
        
    def getFullPath ( self ) :
        return self._fullPath
    
    def getRootPath ( self ):
        return self._rootPath
    
    def getExtension ( self ):
        return self._extension
    
    def exists ( self ):
        self._exists = os.path.exists( self._name )
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
        self._fileName = os.path.basename( self._fullPath )
        self._rootPath, self._extension = os.path.splitext ( self._fullPath )
                
 
class ImageFile ( File ):
     
     _IMAGE_TYPE_PNG = ".png"
     _IMAGE_TYPE_JPG = ".jpg"
     _IMAGE_TYPE_JPEG = ".jpeg"
     _IMAGE_TYPE_GIF = ".gif"
     
     def __init__ ( self, fullPath ):
         super ().__init__ ( fullPath )
         self._isPNG = False
         self._isJPG = False
         self._isGIF = False
         self.__determineImageType__ ()
         
     def isPNG ( self ):
        return self._isPNG
    
     def isJPG ( self ):
        return self.print_jpg
    
     def  isGIF ( self ):
        return self._isGIF
    
     def __determineImageType__ ( self ):
        lowercase_extension = self._extension.lower ()
        if lowercase_extension ==self. _IMAGE_TYPE_PNG :
            self._isPNG = True
        elif lowercase_extension == self._IMAGE_TYPE_JPG or lowercase_extension == self._IMAGE_TYPE_JPEG :
            self._isJPG = True
        elif lowercase_extension == self._IMAGE_TYPE_GIF :
            self._isGIF = True
        
        
        


class Directory ( File ):
    def __init__ (self, fullPath ):
        super ().__init__ ( fullPath )
        self._fileList = []
        self._fullPathFileList = []
        self._numFiles = 0
        
    def getName (self ):
        return self._fullPath
    
    def setName (self, fullPath ):
        super.setFullPath ( fullPath )
        return self.exists ( self )
    
    def exists ( self ) :
        if not os.path.exists( self._fullPath ):
            self.directoryExists = False
        return self.directoryExists
    
    def create (self ):
        return self.diretoryExists
    
    def getFiles ( self ):
        try :
            self._fileList = os.listdir( self._fullPath  )
            self._numFiles = len ( self._fileList )
        except FileNotFoundError :
            self._exists = False;
            self._numFiles = 0;
            self._fileList = []
        return self._fileList
    
    def getFullPathFiles ( self ) :
        self.getFiles ()
        for i in range (self._numFiles ):
            full_path_name = self._fullPath +"\\" + self._fileList [ i ]
            self._fullPathFileList.append ( full_path_name )
        return self._fullPathFileList
    
    def getNumFiles ( self ):
        return self._numFiles
    
    
    
if __name__ == "__main__":
    
    
    file = File ( "comfyu.png" )
    
    if isinstance ( file, ImageFile ) :
        file.isPNG ()
        num_pngs += 1
        
    
    extension = file.getExtension ()
    directory = Directory ("F:\Pictures\OnlyFans\Resource\progress" )
    dir_list = directory.getFullPathFiles ()
    num_files = directory.getNumFiles ()

    print ( num_files )
    name = directory.getFullPath ()
    file.create ()

    file_list = []
    
    for i in range ( num_files ):
        file_list.append ( File ( dir_list [i] ) )
    
    num_pngs = 0
    for i in range ( num_files ):
        print ( file_list [ i ].getFullPath () )
        
        
    exists = file.exists ()
    print ( exists )
    
    
    
    
    
    
    