from descTxtStructure import DescTxtStructure

###
#  Get data textual descriptors data from file and store it in a dictionary, 
# whose key is objectId and values is list of classes of descTxtStructure
###

def getDescTxtData(filePath):
    file_Pointer = open(filePath, 'r')
    ### read each user's data, one line/row at a time
    master_dict = dict()
    for lines in file_Pointer:
        line = lines.split()
        ### get the user/image/location id as key
        key_id = line[0]
        values = []
        ### store all this as key values pairs in a dictionary
        if key_id in master_dict:
            values = master_dict[key_id]
        else:
            master_dict[key_id] = values
        ### store these values in a dict[user/image/location_id] ->  list of tags stored as array of objects
        for i in range(1, len(line), 4):
            values.append(DescTxtStructure(line[i:i+4]))
    return master_dict
