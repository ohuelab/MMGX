from __future__ import division
import operator
from math import log
from time import time
from util.sarpy.SARpytools import *

def collectSubs(structures, grinder):
    if not structures:
        return []
    substructures = []
    for structure in structures:
        substructures.extend(grinder.getFragments(structure))

    # print(' %s\tsubstructures found...' % len(substructures))
    return substructures + collectSubs(substructures, grinder)


def getAlerts(fragments, dataset, target, minHits):
    alerts = []
    for frag in fragments:
        frag.evaluate(dataset, target)
        if frag.LR > 1 and frag.trueMatches >= minHits:
            frag.setTarget(target)
            frag.trainingHits = frag.hits.getCopy()
            frag.absLR = frag.LR
            frag.absPrecision = frag.precision
            alerts.append(frag)

    return alerts


def getRuleset(alerts, dataset, minHits, minLR, minPrecision):
    collection = []
    rules = alerts[:]
    workingset = dataset.getCopy()
    suspectMode = False
    while rules and workingset.tot():
        rules.sort(key=operator.attrgetter('LR', 'recall', 'absLR', 'atoms', 'priority', 'cansmiles'))
        SA = rules.pop()
        if suspectMode:
            checkFreq = lambda alert: alert.trueMatches >= minHits and alert.absLR == float('inf')
        else:
            checkFreq = lambda alert: workingset.totClass(alert.target) and alert.trueMatches >= max(minHits, log(workingset.totClass(alert.target)))
        infrequents = []
        while rules and not checkFreq(SA):
            infrequents.append(SA)
            SA = rules.pop()

        rules.extend(infrequents)
        if SA.LR < minLR or SA.precision < minPrecision:
            if suspectMode:
                break
            suspectMode = True
            continue
        collection.append(SA)
        workingset.discard(SA.hits)
        for nextSA in rules:
            nextSA.hits.discard(SA.hits)
            nextSA.evaluate(workingset, nextSA.target)
            nextSA.workingset = [0, 0]

    for SA in alerts:
        SA.hits = SA.trainingHits.getCopy()

    interferenceCheck(collection)
    return collection


def interferenceCheck(ruleset):
    ruleset = ruleset[:]
    for i, r in enumerate(ruleset):
        r.ID = i + 1

    while ruleset:
        alert = ruleset.pop(0)
        alert.generalizedBy = []
        for other in ruleset:
            if alert.hits.isSubset(other.hits) and alert.target == other.target:
                alert.generalizedBy.append(other.ID)


def loadDataset(path, ext, labDict = None, smilesKey = None, _filter = None):
    print ('\n\nLoading dataset...')
    loader = Loader(path, ext)
    if smilesKey:
        loader.setSmilesHeader(smilesKey)
    if labDict:
        loader.setLabelDict(labDict)
    trainingset = loader.load(_filter)
    print('\n Read %s molecular structures' % trainingset.tot())
    for label in trainingset.getLabels():
        print('', trainingset.totClass(label), label)

    return trainingset


def fragmentize(dataset, minAtoms, maxAtoms, target = None):
    warning = ''
    if target:
        warning = '[%s ONLY]' % target
    print('\n\nFragmenting...\t%s\n' % warning)
    grinder = Grinder(minAtoms, maxAtoms)
    dataset.setup()
    start = time()
    frags = collectSubs(dataset.IsValid(target), grinder)
    fragTime = time()
    print('\nFRAGMENTS: %s' % len(frags))
    print('\nEvaluating fragments on the training set...')
    for frag in sorted(frags, key=operator.attrgetter('atoms')):
        frag.matchDataset(dataset)

    matchTime = time()
    print('\n    -> elapsed time: %.2f seconds' % (time() - start))
    print('         fragmentation %.2f seconds' % (fragTime - start))
    print('              matching %.2f seconds' % (matchTime - fragTime))
    return frags

def extract(dataset, minHits = 3, minLR = 1, minPrecision = None, target = None):
    warning = ''
    if target:
        warning = '[%s ONLY]' % target
    print('\n\nExtracting rules...\t%s' % warning)
    if target:
        targets = (target,)
    else:
        targets = dataset.getLabels()
    alerts = []
    start = time()
    for target in targets:
        subs = dataset.getSubs(target)
        print('\n %s %s substructures' % (len(subs), target))
        subs.difference_update(alerts)
        newAlerts = getAlerts(subs, dataset, target, minHits)
        print('  %s of which are potential alerts' % len(newAlerts))
        alerts.extend(newAlerts)

    rules = getRuleset(alerts, dataset, minHits, minLR, minPrecision)
    if len(targets) > 1:
        print('\n Extracted:')
        counterDict = dict.fromkeys(targets, 0)
        for r in rules:
            counterDict[r.target] += 1

        for k, v in counterDict.iteritems():
            print(' %s\t%s' % (v, k))

    print('\nRULES: %s' % len(rules))
    print('\n -> time: %.2f seconds' % (time() - start))
    return rules


def classifier_test(dataset):
    labels = dataset.getLabels()
    outcomes = labels + ['unknown']
    m = [ [0] * len(outcomes) for i in range(len(labels)) ]
    unpredIndex = len(outcomes) - 1
    for expIndex, activityLabel in enumerate(labels):
        for structure in dataset.getStructures(activityLabel):
            if structure.pred not in labels:
                m[expIndex][unpredIndex] += 1
            else:
                m[expIndex][labels.index(structure.pred)] += 1

    total = dataset.tot()
    correctno = 0
    unpredicted = 0
    for i in range(len(labels)):
        correctno += m[i][i]
        unpredicted += m[i][unpredIndex]

    errors = total - correctno - unpredicted
    print('\n ERROR RATE: %.2f' % (errors / total))
    print(' Unpredicted rate: %.2f' % (unpredicted / total))
    print('\nCONFUSION MATRIX:')
    for c in outcomes:
        print(c[:7] + '\t',)

    print('<-predicted')
    for i, label in enumerate(labels):
        for v in m[i]:
            print(str(v) + '\t',)

        print('%s' % label, '\n',)


def extractor_test(dataset, target):
    targets = dataset.getStructures(target)
    nontargets = dataset.getStructures() - targets
    TP = TN = FP = FN = 0
    for structure in targets:
        if structure.pred == target:
            TP += 1
        else:
            FN += 1

    for structure in nontargets:
        if structure.pred != target:
            TN += 1
        else:
            FP += 1

    accuracy = (TP + TN) / (TP + FN + TN + FP)
    targets = TP + FN
    nontargets = TN + FP
    if not targets:
        sensitivity = 0
    else:
        sensitivity = TP / targets
    if not nontargets:
        specificity = 0
    else:
        specificity = TN / nontargets
    print('\n ACCURACY:\t%.2f' % accuracy)
    print(' sensitivity:\t%.2f' % sensitivity)
    print(' specificity:\t%.2f' % specificity)
    print('\nCONFUSION MATRIX:')
    print('YES\tNO\t<-any alert?')
    print('%s\t%s\tPOSITIVES' % (TP, FN))
    print('%s\t%s\tNEGATIVES' % (FP, TN))


def predict(ruleset, dataset):
    print('\n\nPredicting...')
    if not dataset or dataset.tot() == 0:
        print('\n *** DATASET empty')
        return False
    if not ruleset or len(ruleset) == 0:
        print('\n *** RULESET empty')
        return False
    c = 0
    for structure in dataset.getStructures():
        structure.pred = None
        structure.rule = None
        for SA in sorted(ruleset, key=operator.attrgetter('absLR'), reverse=True):
            if SA.match(structure.mol):
                structure.rule = SA
                structure.pred = structure.rule.target
                c += 1
                break

    print('\n %s structures matched' % c)
    return True


def validate(dataset):
    print('\n\nValidating...')
    targets = set()
    for s in dataset.getStructures():
        if s.pred:
            targets.add(s.pred)

    for target in targets:
        if target not in dataset.getLabels():
            print('\n *** ERROR: RULESET and DATASET are incompatible!')
            print(" There is a '%s' prediction, but the DATASET doesn't have such key..." % target)
            return

    if not targets:
        print('\n*** Unpredicted DATASET')
        return
    if len(targets) == 1:
        target = targets.pop()
        print('\n Binary classification:')
        print('  %s = POSITIVE' % target)
        print('  otherwise = NEGATIVE')
        extractor_test(dataset, target)
    elif len(targets) > 1:
        print('\n Multiclass classification:')
        for target in targets:
            print('  %s' % target)

        classifier_test(dataset)


def loadSmarts(filename):
    f = open(filename)
    ruleset = []
    f.readline()
    for line in f.readlines():
        line = line.rstrip('\n')
        smarts, target, absLR = line.split('\t')
        structure = Fragment(smarts)
        structure.setTarget(target)
        structure.absLR = float(absLR)
        ruleset.append(structure)

    f.close()
    print('\n\n%s RULES have been loaded' % len(ruleset))
    return ruleset


def saveSmarts(ruleset, filename):
    f = open(filename, 'w')
    sep = '\t'
    header = sep.join(['SMARTS', 'Target', 'Training LR'])
    f.write(header + '\n')
    for rule in ruleset:
        lr = '%.2f' % rule.absLR
        if rule.absLR == float('inf'):
            lr = 'inf'
        row = sep.join([rule.smartsString, str(rule.target), lr])
        f.write(row + '\n')

    f.close()
    print('\n\n%s RULES have been saved' % len(ruleset))


def savePredictions(dataset, filename, keys = None, sep = '\t'):
    sep = str(sep)
    if not keys:
        keys = []
    header = sep.join(['SMILES',
     'Prediction',
     'Training LR',
     'SMARTS'] + keys)
    f = open(filename, 'w')
    f.write(header + '\n')
    for structure in dataset.getSorted():
        smarts = lr = ''
        if structure.rule:
            smarts = structure.rule.smartsString
            lr = '%.2f' % structure.rule.absLR
        attributes = [structure.smiles,
         str(structure.pred),
         lr,
         smarts]
        if keys:
            attributes += [ structure.data[key] for key in keys ]
        row = sep.join(attributes)
        f.write(row + '\n')

    f.close()
    print('\n\nPredictions saved')


def debug(ruleset):
    out = open('output.csv', 'w')
    out.write('SA_ID, SMARTS, activity, LR, absLR, workingset, generalized by, relativeTP, relativeFP, True_Matches, False_Matches, True_Mol_ID, False_Mol_ID, True_SMILES, False_SMILES\n')
    for f in ruleset:
        hitIDs = []
        hitSmiles = []
        try:
            if f.workingset:
                f.working = '(%s %s)' % (f.workingset[0], f.workingset[1])
        except:
            f.working = 'ALL'

        for hit in f.hits.getStructures(f.target):
            hitIDs.append(hit.data['Mol_ID'])
            hitSmiles.append(hit.smiles)

        errorIDs = []
        errorSmiles = []
        for hit in f.hits.getStructures() - f.hits.getStructures(f.target):
            errorIDs.append(hit.data['Mol_ID'])
            errorSmiles.append(hit.smiles)

        row = {}
        row['generalized'] = str(f.generalizedBy)
        row['hitsID'] = str(hitIDs)
        row['errID'] = str(errorIDs)
        row['hitsmi'] = str(hitSmiles)
        row['errsmi'] = str(errorSmiles)
        for k, string in row.iteritems():
            string = string.replace(',', ';')
            row[k] = string.replace("'", '')

        out.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (f.ID,
         f.smartsString,
         f.target,
         f.LR,
         f.absLR,
         f.working,
         row['generalized'],
         f.trueMatches,
         f.falseMatches,
         len(hitIDs),
         len(errorIDs),
         row['hitsID'],
         row['errID'],
         row['hitsmi'],
         row['errsmi']))

    out.close()


def go():
    f = open('output.txt', 'w')
    for r in ruleset:
        pos = set(r.hits.getSorted(r.target))
        neg = set(r.hits.getSorted()) - pos
        f.write('\n')
        f.write(str(r.ID) + '\t' + r.smartsString + '\n')
        f.write('PPV= ' + str(r.absPrecision) + '\n')
        f.write('generalized by: ' + str(r.generalizedBy) + '\n')
        f.write(' TP: ' + str(len(pos)) + '\n')
        for tp in pos:
            f.write('  ' + tp.data['Mol_ID'] + '\t' + tp.smiles + '\n')

        f.write('\n FP: ' + str(len(neg)) + '\n')
        for fp in neg:
            f.write('  ' + fp.data['Mol_ID'] + '\t' + fp.smiles + '\n')

    f.close()
