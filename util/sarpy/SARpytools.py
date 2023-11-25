from __future__ import division
import operator
from string import digits, ascii_letters
from math import log
from csv import reader
from openbabel import openbabel
from openbabel import OBConversion, obErrorLog
# from pybel import Smarts, readfile, readstring
from openbabel import pybel as pb
obErrorLog.SetOutputLevel(0)
Break = 'x'
WildCard = '[*]'
Asterisk = '*'
WildCards = [Break, WildCard, Asterisk]
TwoCharsElements = ['Br', 'Cl']
OneCharElements = ['B', 'C', 'N', 'O', 'P', 'S', 'F', 'I']
AromaticElements = ['c', 'n', 'o', 's', 'p']
Bonds = ['-', '=', '#', ':']
SplittingChars = OneCharElements + [ e[0] for e in TwoCharsElements ] + AromaticElements

class Grinder:

    def __init__(self, minAtoms, maxAtoms):
        self.minAtoms = minAtoms
        self.maxAtoms = maxAtoms
        self.fragDict = {}

    def _checkSize(self, smilesFrag):
        smiles = smilesFrag
        c = 0
        s = ''
        inBrackets = False
        for wildcard in WildCards:
            smiles = smiles.replace(wildcard, '')

        for char in smiles:
            if char == '[':
                inBrackets = True
                c += 1
                continue
            if char == ']':
                inBrackets = False
                continue
            if not inBrackets and char in ascii_letters:
                s = s + char

        for chars in TwoCharsElements:
            c += s.count(chars)
            s = s.replace(chars, '')

        atoms = len(s) + c
        return self.minAtoms <= atoms <= self.maxAtoms

    def _checkLevel(self, smilesFrag, smiles):
        return smilesFrag.count(Asterisk) >= smiles.count(Asterisk)

    def _checkCycles(self, smilesFrag):
        for n in digits:
            if smilesFrag.count(n) % 2:
                return False

        return True

    def getFragments(self, structure):

        def split(index):
            rearChar = smiles[index - 1]
            if rearChar in Bonds:
                bond = rearChar
            else:
                bond = ''
            if parenthesesStack:
                closingParenthesesIndex = parenthesesStack[-1]
                leftChip = smiles[:index]
                rightChip = smiles[closingParenthesesIndex:]
                middleChip = smiles[index:closingParenthesesIndex]
                head = leftChip + Break + rightChip
                tail = Break + bond + middleChip
            else:
                leftChip = smiles[:index]
                rightChip = smiles[index:]
                head = leftChip + Break
                tail = Break + bond + rightChip
            return (head, tail)

        substructures = []
        parenthesesStack = []
        inBrackets = False
        smiles = structure.smiles
        currPos = len(smiles)
        while currPos > 1:
            splitHere = False
            currPos -= 1
            char = smiles[currPos]
            if inBrackets and char != '[':
                continue
            if char == ']':
                inBrackets = True
            elif char == '[':
                inBrackets = False
                splitHere = True
            elif char == ')':
                parenthesesStack.append(currPos)
            elif char == '(':
                parenthesesStack.pop()
            elif char in SplittingChars:
                if char not in AromaticElements or smiles[currPos + 1] in digits:
                    splitHere = True
            if splitHere:
                for smilesFrag in split(currPos):
                    if not self._checkLevel(smilesFrag, smiles):
                        continue
                    if not self._checkCycles(smilesFrag):
                        break
                    if not self._checkSize(smilesFrag):
                        continue
                    try:
                        newFrag = Fragment(smilesFrag)
                    except:
                        break

                    if newFrag.cansmiles in self.fragDict:
                        structure.connect(self.fragDict[newFrag.cansmiles])
                    else:
                        structure.connect(newFrag)
                        newFrag.calcfp()
                        substructures.append(newFrag)
                        self.fragDict[newFrag.cansmiles] = newFrag

        return substructures


class Converter:

    def __init__(self, can = False):
        self._OBconverter = OBConversion()
        self._OBconverter.SetOutFormat('smi')
        options = 'in'
        if can:
            options += 'c'
        self._OBconverter.SetOptions(options, OBConversion.OUTOPTIONS)

    def getSmiles(self, mol):
        smiles = self._OBconverter.WriteString(mol.OBMol)[:-1]
        if '@' in smiles:
            smiles = self._OBconverter.WriteString(mol.OBMol)[:-1]
        return smiles


class Structure:
    _converter = Converter()
    def __init__(self, mol):
        self.mol = mol
        self.smiles = Structure._converter.getSmiles(self.mol)
        self._fingerprint = set(self.mol.calcfp().bits)
        self.data = None

    def setup(self):
        self._substructures = set()

    def updateData(self, dictionary):
        self.data = dictionary

    def getData(self, key):
        return self.data[key]

    def addSubstructure(self, sub):
        self._substructures.add(sub)

    def getSubstructures(self):
        return self._substructures

    def connect(self, child):
        pass


class Fragment:
    _converter = Converter(can=True)
    _nhString = '[nH]'
    _nh = pb.Smarts(_nhString)
    _biphenylfp = set(pb.readstring('smi', 'c1ccccc1c1ccccc1').calcfp().bits)

    def __init__(self, smilesFrag):
        self.smiles = smilesFrag.replace(Break, Asterisk)
        self._molSmiles = self._removedAtom(self.smiles, Asterisk)
        self.mol = pb.readstring('smi', self._molSmiles)
        self.atoms = len(self.mol.atoms) - self._molSmiles.count(WildCard) - self._molSmiles.count('H')
        self.smartsString = self._removedAtom(self._molSmiles, WildCard)
        self._smarts = pb.Smarts(self.smartsString)
        if not self.match(self.mol) or len(Fragment._nh.findall(self.mol)) != self.smartsString.count(Fragment._nhString):
            self.smiles = smilesFrag.replace(Break, WildCard)
            self._molSmiles = self._removedAtom(self.smiles, Asterisk)
            self.mol = pb.readstring('smi', self._molSmiles)
        self.cansmiles = Fragment._converter.getSmiles(self.mol)
        self._fingerprint = None
        self.target = None
        self._childs = set()

    def _removedAtom(self, smiles, atom):
        removeChars = []
        for bond in Bonds:
            removeChars.extend((atom + bond, bond + atom))

        removeChars.extend((atom, '()'))
        for char in removeChars:
            smiles = smiles.replace(char, '')

        return smiles.replace('[]', WildCard)

    def connect(self, child):
        self._childs.add(child)

    def calcfp(self):
        if self._molSmiles.count(WildCard) == 0:
            fp = set(self.mol.calcfp().bits)
            if not fp.issuperset(Fragment._biphenylfp):
                self._fingerprint = fp
        del self.mol

    def matchDataset(self, dataset):
        self.hits = Dataset(dataset.labels)
        searchingset = dataset
        if self._childs:
            searchingset = min([ child.hits for child in self._childs ] + [dataset], key=operator.methodcaller('tot'))
        for label in dataset.classes:
            hits = self._find(searchingset.getStructures(label))
            self.hits.populate(hits, label)

        for structure in self.hits.getStructures():
            structure.addSubstructure(self)

    def setTarget(self, label):
        self.target = label

    def evaluate(self, dataset, target):
        '''
        ACC = self.precision
        PR  = self.pr
        IG = self.ig
        '''
        def informationgain(p,n,tp,tn,a,i):
            '''
            @param p : number of predicted positive items
            @param n : number of predicted negative items
            @param tp: number of true positive items
            @param tn: number of true negative items
            @param a: number of real postive items
            @param i: number of real negative items
            '''
            def entropy(p1,p2):
                    log2 = lambda x:log(x)/log(2)
                    if p1 == 0 or p1 == 1:
                        return 0
                    return -p1*log2(p1)-p2*log2(p2)
            if p == 0 or n == 0:
                return 0
            HxP1= a * 1.0 / (a+i)
            Hx = entropy(HxP1,1-HxP1)
            HtP1 = tp * 1.0 / p
            Ht = entropy(HtP1,1-HtP1)
            HtbP1 = tn * 1.0 / n
            Htb = entropy(HtbP1,1-HtbP1)
            Pt = p * 1.0 / (p+n)
            HT = Pt * Ht + (1-Pt) * Htb
            return Hx-HT
        self.LR = self.precision = self.recall = None
        targets = dataset.totClass(target)
        nontargets = dataset.tot() - targets
        self.trueMatches = T = self.hits.totClass(target)
        self.falseMatches = F = self.hits.tot() - self.trueMatches
        self.priority = targets > nontargets
        if not targets:
            return
        if T == 0:
            return
        if F == 0:
            self.LR = float('inf')
            self.precision = 1
        else:
            self.LR = T / F * (nontargets / targets)
            self.precision = T / (T + F)
        n = dataset.tot()
        tn = nontargets - F
        self.ig = informationgain(T+F,n-T-F,T,tn,targets,nontargets)
        self.pr = (T+F) / n
        self.recall = T / targets

    def match(self, mol):
        return self._smarts.obsmarts.Match(mol.OBMol, True)

    def _find(self, structures):
        hits = []
        for structure in structures:
            if self._fingerprint and not self._fingerprint.issubset(structure._fingerprint):
                continue
            if self.match(structure.mol):
                hits.append(structure)

        return hits

class Filter:

    def __init__(self, key = None, value = None, op = operator.eq):
        self.conditions = []
        if key and value != None:
            self.addCondition(key, value, op)

    def __str__(self):
        return 'Filter: %s %s %s' % (self.op, self.key, self.value)

    def addCondition(self, key, value, op = operator.eq):
        self.key = key
        self.value = value
        self.op = op
        if type(value) == str:
            self.conditions.append(lambda molData: op(molData[key], value))
        else:
            self.conditions.append(lambda molData: op(float(molData[key]), value))

    def evaluate(self, molData):
        for cond in self.conditions:
            try:
                if not cond(molData):
                    return False
            except:
                    return False
        return True


class Filter_OR(Filter):

    def __init__(self):
        self.conditions = []

    def evaluate(self, molData):
        for cond in self.conditions:
            try:
                if cond(molData):
                    return True
            except:
                continue


class Loader:
    defaultLabel = 'ALL'

    def __init__(self, path, ext):
        self.path = path
        self.ext = ext
        self.labels = [self.defaultLabel]
        self.labDict = None
        self.smilesHeader = None

    def setSmilesHeader(self, smilesHeader):
        self.smilesHeader = smilesHeader

    def setLabelDict(self, labDict):
        self.labDict = labDict
        self.labels = labDict.keys()

    def _getLabel(self, data):
        if not self.labDict:
            return self.defaultLabel
        for label, _filter in self.labDict.iteritems():
            if _filter.evaluate(data):
                return label.replace('\t', '_')

    def load(self, _filter = None):
        load = operator.methodcaller('_read' + self.ext, _filter)
        return load(self)

    def _readsdf(self, _filter = None):
        dataset = Dataset(self.labels)
        for mol in pb.readfile('sdf', self.path):
            if mol.atoms:
                data = mol.data
                if _filter and not _filter.evaluate(data):
                    continue
                structure = Structure(mol)
                label = self._getLabel(data)

                if label:
                    structure.updateData(data)
                    dataset.add(structure, label)

        return dataset

    def _readcsv(self, _filter = None):
        errors = 0
        if not self.smilesHeader:
            raise Exception('Missing SMILES header')
        dataset = Dataset(self.labels)
        firstRow = True
        for row in reader(open(self.path)):
            if firstRow:
                headers = row
                firstRow = False
                continue
            data = dict(zip(headers, row))
            if _filter and not _filter.evaluate(data):
                continue
            try:
                mol = pb.readstring('smi', data[self.smilesHeader])
            except IOError as error:
                print(error)
                errors += 1
                continue

            structure = Structure(mol)
            label = self._getLabel(data)
            print(label)
            if label:
                structure.updateData(data)
                dataset.add(structure, label)

        if errors:
            print ('\n*** SMILES error: %s structure(s) discarded\n' % errors)
        return dataset


class Dataset:

    def __init__(self, labels):
        self.labels = labels
        self.classes = {}
        self.counter = 0
        for label in labels:
            self.classes[label] = set()

    def setup(self):
        for structure in self.getStructures():
            structure.setup()

    def add(self, structure, label):
        self.counter += 1
        structure.ID = self.counter
        self.classes[label].add(structure)

    def populate(self, structures, label):
        for structure in structures:
            self.add(structure, label)

    def discard(self, other):
        for label in other.getLabels():
            self.getStructures(label).difference_update(other.getStructures(label))

    def isSubset(self, other):
        issubset = True
        for label in self.getLabels():
            issubset = issubset and self.getStructures(label).issubset(other.getStructures(label))

        return issubset

    def getCopy(self):
        dataset = Dataset(self.labels)
        for label in self.getLabels():
            dataset.populate(self.getStructures(label).copy(), label)

        return dataset

    def getStructures(self, label = None):
        if label in self.getLabels():
            return self.classes[label]
        if not label:
            structures = set()
            for label in self.getLabels():
                structures.update(self.getStructures(label))

            return structures

    def getSorted(self, label = None):
        return sorted(self.getStructures(label), key=operator.attrgetter('ID'))

    def getSubs(self, label):
        subs = set()
        for structure in self.getStructures(label):
            subs.update(structure.getSubstructures())

        return subs

    def totClass(self, label):
        return len(self.getStructures(label))

    def tot(self):
        tot = 0
        for label in self.getLabels():
            tot += self.totClass(label)

        return tot

    def getLabels(self):
        return self.classes.keys()
