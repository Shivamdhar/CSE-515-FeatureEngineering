from constants import *
from descTxtStructure import DescTxtStructure

class TxtTermStructure:
    
    def __init__(self):
        self.master_dict = dict()

###
#  Get data textual descriptors data for users from file and store it in dictionaries
###

    def loadUsersData(self):
        self.master_dict = dict()
        self.getDescTxtData(TEXT_DESCRIPTORS_PATH+'devset_textTermsPerUser.txt', 'users')
        
###
#  Get data textual descriptors data for images from file and store it in dictionaries
###

    def loadImageData(self):
        self.master_dict = dict()
        self.getDescTxtData(TEXT_DESCRIPTORS_PATH+'devset_textTermsPerImage.txt', 'image')
        
###
#  Get data textual descriptors datafor location from file and store it in dictionaries
###

    def loadLocationData(self):
        self.master_dict = dict()
        self.getDescTxtData(TEXT_DESCRIPTORS_PATH+'devset_textTermsPerPOI.wFolderNames.txt', 'location')
        
###
#  Get data textual descriptors data for all from file and store it in dictionaries
###

    def loadAllTextualData(self):
        self.master_dict = dict()
        self.getDescTxtData(TEXT_DESCRIPTORS_PATH+'devset_textTermsPerPOI.wFolderNames.txt', 'location')
        self.getDescTxtData(TEXT_DESCRIPTORS_PATH+'devset_textTermsPerImage.txt', 'image')
        self.getDescTxtData(TEXT_DESCRIPTORS_PATH+'devset_textTermsPerUser.txt', 'users')
                

###
#  Get data textual descriptors data from file and store it in a dictionary, 
# whose key is objectId and values is list of objects of type descTxtStructure
###

    def getDescTxtData(self, filePath, data_type):
        file_Pointer = open(filePath, 'r')
        ### read each user's data, one line/row at a time
        for lines in file_Pointer:
            line = lines.split()
            ### get the user/image/location id as key
            key_id = line[0]
            values = []
            ### get index of first word starting with quotes, this is done to 
            ### avoid reading name without spaces in location file
            k = [line.index(x) for x in line[:20] if '"' in x]
            k = k[0]
            ### store all this as key values pairs in a dictionary
            if key_id in self.master_dict:
                values = self.master_dict[key_id]
            else:
                self.master_dict[key_id] = values
            ### store these values in a dict[user/image/location_id] -> 
            ### list of tags stored as array of objects
            for i in range(k, len(line), 4):
                values.append(DescTxtStructure(line[i:i+4], data_type))


###
#  returns an array of terms for a given id (as String), if present in data else returns empty array
#
###

    def getTerms(self, id):
        if id not in self.master_dict:
            print("given id: ", id, " not found")
            return []
        return [texDescriptor.term for texDescriptor in self.master_dict[id] ] 
        