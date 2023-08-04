"""
code modified from the wmh challenge metrics github

I should put their copywrite notice here I think

"""

import SimpleITK as sitk
import numpy as np
import torch
from tqdm import tqdm
import scipy

def getDSC(testImage, resultImage):    
        """Compute the Dice Similarity Coefficient."""
        # testArray   = sitk.GetArrayFromImage(testImage).flatten()
        # resultArray = sitk.GetArrayFromImage(resultImage).flatten()
        testArray = testImage.reshape(-1).cpu().numpy()
        resultArray = resultImage.reshape(-1).cpu().numpy()

        # similarity = 1.0 - dissimilarity
        return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 


def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""

    testImage = sitk.GetImageFromArray(testImage)
    resultImage = sitk.GetImageFromArray(resultImage)
    testImage.SetSpacing((1.,1.,3.))
    resultImage.SetSpacing((1.,1.,3.))

    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float('nan')

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage   = sitk.BinaryErode(testImage, (1,1,0) )
    eResultImage = sitk.BinaryErode(resultImage, (1,1,0) )

    hTestImage   = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)

    hTestArray   = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)   

    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
    resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]


    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):    
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)    

    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))


def getLesionDetection(testImage, resultImage):    
    """Lesion detection metrics, both recall and F1."""
    testImage = sitk.GetImageFromArray(testImage)
    resultImage = sitk.GetImageFromArray(resultImage)

    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()    
    ccFilter.SetFullyConnected(True)

    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)    
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))

    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)

    # recall = (number of detected WMH) / (number of true WMH) 
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH

    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))

    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)

    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return recall, f1    


def getAVD(testImage, resultImage):   
    """Volume statistics."""
    testImage = sitk.GetImageFromArray(testImage)
    resultImage = sitk.GetImageFromArray(resultImage)

    # Compute statistics of both images
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()

    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)

    return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100

def do_challenge_metrics(testImage, resultImage, print_results=False):
    """Main function"""
    dsc = getDSC(testImage, resultImage)
    try:
        h95 = getHausdorff(testImage, resultImage)
    except:
        h95 = 100
    avd = getAVD(testImage, resultImage)    
    recall, f1 = getLesionDetection(testImage, resultImage)    

    if print_results:
        print('Dice',                dsc,       '(higher is better, max=1)')
        print('HD',                  h95, 'mm',  '(lower is better, min=0)')
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,       '(higher is better, max=1)')
        print('Lesion F1',            f1,       '(higher is better, max=1)')

    return dsc, h95, avd, recall, f1

def per_model_chal_stats(preds3d, ys3d, do_argmax=True):
    print("warning: using spacing 1.,1.,3. for HD95 score")
    stats = []
    for i in tqdm(range(len(ys3d)), position=0, leave=True):
        if do_argmax:
            ind_stats = do_challenge_metrics(ys3d[i].type(torch.long), preds3d[i].argmax(dim=1).type(torch.long))
        else:
            ind_stats = do_challenge_metrics(ys3d[i].type(torch.long), preds3d[i].type(torch.long))
        stats.append(ind_stats)

    tstats = torch.Tensor(stats)
    dices = tstats[:,0]
    hd95s = tstats[:,1]
    avds = tstats[:,2]
    recalls = tstats[:,3]
    f1s = tstats[:,4]

    data = {"dice":dices, "hd95":hd95s, "avd":avds, "recall":recalls, "f1":f1s}

    return data

def challenge_results_combined_lists(data, loss_name):
    """
    data is the dictionary returned by the evaluation code above that returns a series of values from the challenge evaluation code, per domain.
    """
    df = {"loss_name":[], "dice":[], "hd95":[], "avd":[], "recall":[], "f1":[]}
    dice = []
    f1 = []
    avd = []
    hd95 = []
    recall = []
    for dd in data:
        dice.extend(dd['dice'].tolist())
        hd95.extend(dd['hd95'].tolist())
        avd.extend(dd['avd'].tolist())
        recall.extend(dd['recall'].tolist())
        f1.extend(dd['f1'].tolist())
    df['dice'].extend(dice)
    df['f1'].extend(f1)
    df['avd'].extend(avd)
    df['hd95'].extend(hd95)
    df['recall'].extend(recall)
    df['loss_name'].extend([loss_name for _ in range(len(dice))])
    
    return df


def create_combined_df(results):
    """
    results is a dictionary of results from my evaluation code,
    for which key: performance results returns a list of challenge metric results
    computed from per_model_chal_stats method, one per domain.
    """
    df = {"loss_name":[], "dice":[], "hd95":[], "avd":[], "recall":[], "f1":[]}
    for data in results:
        loss_name = data["loss"] + "_" + data["lr"]
        dice = []
        f1 = []
        avd = []
        hd95 = []
        recall = []
        for dd in data['performance_results']:
            dice.extend(dd['dice'].tolist())
            hd95.extend(dd['hd95'].tolist())
            avd.extend(dd['avd'].tolist())
            recall.extend(dd['recall'].tolist())
            f1.extend(dd['f1'].tolist())
        df['dice'].extend(dice)
        df['f1'].extend(f1)
        df['avd'].extend(avd)
        df['hd95'].extend(hd95)
        df['recall'].extend(recall)
        df['loss_name'].extend([loss_name for _ in range(len(dice))])
    
    return df