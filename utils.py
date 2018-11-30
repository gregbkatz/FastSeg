import pdb

def parseLine(line):
    words = line.split(',')
    class_dict = {}
    class_id = int(words[0])
    class_dict['id'] = class_id
    class_dict['name'] = words[1].strip()
    class_dict['coco_id'] = int(words[2])
    class_dict['count'] = int(words[3])
    class_dict['frequency'] = float(words[4])
    class_dict['weight'] = float(words[5])
    return class_id, class_dict

class Classes:
    def __init__(self, infile):
        self.classes = {}
        self.name2id = {}
        self.cocoid2id = {}
    
        with open(infile, "r") as f:
            for line in f:
                if line[0] != '#':
                    class_id, class_dict = parseLine(line)
                    self.classes[class_id] = class_dict
                    self.name2id[class_dict['name']] = class_id
                    self.cocoid2id[class_dict['coco_id']] = class_id

    def getList(self, key):
        return [x[1][key] for x in self.classes.items()]

    def getWeights(self):
        return self.getList('weight')

    def getNames(self):
        return self.getList('name')

    def getId(self, class_id):
        return self.classes[class_id]
  
    def getName(self, name): 
        return self.classes[self.name2id[name]]

    def getCocoId(self, coco_id):
        return self.classes[self.cocoid2id[coco_id]]

    def print(self):
        for key in self.classes:
            print(self.classes[key])

