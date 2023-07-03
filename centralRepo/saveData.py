# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
Data processing functions for EEG data
@author: Ravikiran Mane
"""
import numpy as np
import mne
from scipy.io import loadmat, savemat
import os
import pickle
import csv
from shutil import copyfile
import sys
import resampy
import shutil
import urllib.request as request
from contextlib import closing

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))  # To load all the relevant files
from eegDataset import eegDataset
import transforms


def parseBci42aFile(dataPath, labelPath, epochWindow=[0, 4], chans=list(range(22))):
    '''
    Parse the bci42a data file and return an epoched data. 

    Parameters
    ----------
    dataPath : str
        path to the gdf file.
    labelPath : str
        path to the labels mat file.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list : channels to select from the data. 

    Returns
    -------
    data : an EEG structure with following fields:
        x: 3d np array with epoched EEG data : chan x time x trials
        y: 1d np array containing trial labels starting from 0
        s: float, sampling frequency
        c: list of channels - can be list of ints.
    '''
    eventCode = [2]  # start of the trial at t=0
    fs = 250
    offset = 2

    # load the gdf file using MNE
    raw_gdf = mne.io.read_raw_gdf(dataPath, stim_channel="auto")
    raw_gdf.load_data()
    gdf_events = mne.events_from_annotations(raw_gdf)[0][:, [0, 2]].tolist()
    eeg = raw_gdf.get_data()

    # drop channels
    if chans is not None:
        eeg = eeg[chans, :]

    # Epoch the data
    events = [event for event in gdf_events if event[1] in eventCode]
    y = np.array([i[1] for i in events])
    epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + offset * fs
    x = np.stack([eeg[:, epochInterval + event[0]] for event in events], axis=2)

    # Multiply the data with 1e6
    x = x * 1e6

    # Load the labels
    y = loadmat(labelPath)["classlabel"].squeeze()
    # change the labels from [1-4] to [0-3]
    y = y - 1

    data = {'x': x, 'y': y, 'c': np.array(raw_gdf.info['ch_names'])[chans].tolist(), 's': fs}
    return data


def parseBci42bFile(dataPath, labelPath, epochWindow=[0, 4], chans=list(range(3))):

    # start of the trial at t=0
    if int(dataPath[-6:-5]) == 1 or int(dataPath[-6:-5]) == 2:
        eventCode = [2]
    else:
        eventCode = [3]
    fs = 250
    offset = 3

    # load the gdf file using MNE
    raw_gdf = mne.io.read_raw_gdf(dataPath, stim_channel="auto")
    raw_gdf.load_data()
    gdf_events = mne.events_from_annotations(raw_gdf)[0][:, [0, 2]].tolist()
    eeg = raw_gdf.get_data()

    # drop channels
    if chans is not None:
        eeg = eeg[chans, :]

    # Epoch the data
    events = [event for event in gdf_events if event[1] in eventCode]
    y = np.array([i[1] for i in events])
    epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + offset * fs
    x = np.stack([eeg[:, epochInterval + event[0]] for event in events], axis=2)

    # Multiply the data with 1e6
    x = x * 1e6

    # Load the labels
    y = loadmat(labelPath)["classlabel"].squeeze()
    # change the labels from [1-4] to [0-3]
    y = y - 1

    data = {'x': x, 'y': y, 'c': np.array(raw_gdf.info['ch_names'])[chans].tolist(), 's': fs}
    return data


def parseBci42aDataset(datasetPath, savePath,
                       epochWindow=[0, 4], chans=list(range(22)), verbos=False):
    '''
    Parse the BCI comp. IV-2a data in a MATLAB formate that will be used in the next analysis

    Parameters
    ----------
    datasetPath : str
        Path to the BCI IV2a original dataset in gdf formate.
    savePath : str
        Path on where to save the epoched eeg data in a mat format.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list : channels to select from the data.

    Returns
    -------
    None.
    The dataset will be saved at savePath.

    '''
    subjects = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
    test_subjects = ['A01E', 'A02E', 'A03E', 'A04E', 'A05E', 'A06E', 'A07E', 'A08E', 'A09E']
    subAll = [subjects, test_subjects]
    subL = ['s', 'se']  # s: session 1, se: session 2 (session evaluation)

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSubs, subs in enumerate(subAll):
        for iSub, sub in enumerate(subs):
            if not os.path.exists(os.path.join(datasetPath, sub + '.mat')):
                raise ValueError('The BCI-IV-2a original dataset doesn\'t exist at path: ' +
                                 os.path.join(datasetPath, sub + '.mat') +
                                 ' Please download and copy the extracted dataset at the above path ' +
                                 ' More details about how to download this data can be found in the Instructions.txt file')

            print('Processing subject No.: ' + subL[iSubs] + str(iSub + 1).zfill(3))
            data = parseBci42aFile(os.path.join(datasetPath, sub + '.gdf'),
                                   os.path.join(datasetPath, sub + '.mat'),
                                   epochWindow=epochWindow, chans=chans)
            savemat(os.path.join(savePath, subL[iSubs] + str(iSub + 1).zfill(3) + '.mat'), data)


def parseBci42bDataset(datasetPath, savePath,
                       epochWindow=[0, 4], chans=list(range(3)), verbos=False):
    #"start event = 2 left 3 right 4"
    s1t = ['B0101T', 'B0102T', 'B0103T']
    s1e = ['B0104E', 'B0105E']
    s2t = ['B0201T', 'B0202T', 'B0203T']
    s2e = ['B0204E', 'B0205E']
    s3t = ['B0301T', 'B0302T', 'B0303T']
    s3e = ['B0304E', 'B0305E']
    s4t = ['B0401T', 'B0402T', 'B0403T']
    s4e = ['B0404E', 'B0405E']
    s5t = ['B0501T', 'B0502T', 'B0503T']
    s5e = ['B0504E', 'B0505E']
    s6t = ['B0601T', 'B0602T', 'B0603T']
    s6e = ['B0604E', 'B0605E']
    s7t = ['B0701T', 'B0702T', 'B0703T']
    s7e = ['B0704E', 'B0705E']
    s8t = ['B0801T', 'B0802T', 'B0803T']
    s8e = ['B0804E', 'B0805E']
    s9t = ['B0901T', 'B0902T', 'B0903T']
    s9e = ['B0904E', 'B0905E']
    subjects = [s1t, s2t, s3t, s4t, s5t, s6t, s7t, s8t, s9t]
    test_subjects = [s1e, s2e, s3e, s4e, s5e, s6e, s7e, s8e, s9e]
    subAll = [subjects, test_subjects]
    subL = ['s', 'se']  # s: session 1, se: session 2 (session evaluation)

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSubs, subs in enumerate(subAll):
        #[subjects, test_subject]
        for iSub, sub in enumerate(subs):
            subData = {}
            #[s1t, s2t, s3t, s4t, s5t, s6t, s7t, s8t, s9t],[s1e, s2e, s3e, s4e, s5e, s6e, s7e, s8e, s9e]
            for iS, su in enumerate(sub):
                #['B0101T']
                if not os.path.exists(os.path.join(datasetPath, su + '.mat')):
                    raise ValueError('The BCI-IV-2a original dataset doesn\'t exist at path: ' +
                                     os.path.join(datasetPath, su + '.mat') +
                                     ' Please download and copy the extracted dataset at the above path ' +
                                     ' More details about how to download this data can be found in the Instructions.txt file')

                print('Processing subject No.: ' + subL[iSubs] + su[1:5])
                data = parseBci42bFile(os.path.join(datasetPath, su + '.gdf'),
                                       os.path.join(datasetPath, su + '.mat'),
                                       epochWindow=epochWindow, chans=chans)
                if subData:
                    subData['x'] = np.concatenate((subData['x'], data['x']), axis=2)
                    subData['y'] = np.concatenate((subData['y'], data['y']), axis=0)
                else:
                    subData = data

            name = subL[iSubs] + sub[0][2:3].zfill(3)
            savemat(os.path.join(savePath, name + '.mat'), subData)


def fetchAndParseKoreaFile(dataPath, url=None, epochWindow=[0, 4],
                           chans=[7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20],
                           downsampleFactor=4):
    '''
    Parse one subjects EEG dat from Korea Uni MI dataset.

    Parameters
    ----------
    dataPath : str
        math to the EEG datafile EEG_MI.mat.
        if the file doesn't exists then it will be fetched over FTP using url
    url : str, optional
        FTP URL to fetch the data from. The default is None.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list
        channels to select from the data.
    downsampleFactor  : int
        Data down-sample factor

    Returns
    -------
    data : a eeg structure with following fields:
        x: 3d np array with epoched eeg data : chan x time x trials
        y: 1d np array containing trial labels starting from 0
        s: float, sampling frequency
        c: list of channels - can be list of ints or channel names. .

    '''

    eventCode = [1, 2]  # start of the trial at t=0
    s = 1000
    offset = 0

    # check if the file exists or fetch it over ftp
    if not os.path.exists(dataPath):
        if not os.path.exists(os.path.dirname(dataPath)):
            os.makedirs(os.path.dirname(dataPath))
        print('fetching data over ftp: ' + dataPath)
        with closing(request.urlopen(url)) as r:
            with open(dataPath, 'wb') as f:
                shutil.copyfileobj(r, f)

    # read the mat file:
    try:
        data = loadmat(dataPath)
    except:
        print('Failed to load the data. retrying the download')
        with closing(request.urlopen(url)) as r:
            with open(dataPath, 'wb') as f:
                shutil.copyfileobj(r, f)
        data = loadmat(dataPath)

    x = np.concatenate((data['EEG_MI_train'][0, 0]['smt'], data['EEG_MI_test'][0, 0]['smt']), axis=1).astype(np.float32)
    y = np.concatenate((data['EEG_MI_train'][0, 0]['y_dec'].squeeze(), data['EEG_MI_test'][0, 0]['y_dec'].squeeze()),
                       axis=0).astype(int) - 1
    c = np.array([m.item() for m in data['EEG_MI_train'][0, 0]['chan'].squeeze().tolist()])
    s = data['EEG_MI_train'][0, 0]['fs'].squeeze().item()
    del data

    # extract the requested channels:
    if chans is not None:
        x = x[:, :, np.array(chans)]
        c = c[np.array(chans)]

    # down-sample if requested .
    if downsampleFactor is not None:
        xNew = np.zeros((int(x.shape[0] / downsampleFactor), x.shape[1], x.shape[2]), np.float32)
        for i in range(x.shape[2]):  # resampy.resample cant handle the 3D data.
            xNew[:, :, i] = resampy.resample(x[:, :, i], s, s / downsampleFactor, axis=0)
        x = xNew
        s = s / downsampleFactor

    # change the data dimensions to be in a format: Chan x time x trials
    x = np.transpose(x, axes=(2, 0, 1))

    return {'x': x, 'y': y, 'c': c, 's': s}


def parseKoreaDataset(datasetPath, savePath, epochWindow=[0, 4],
                      chans=[7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20],
                      downsampleFactor=4, verbos=False):
    '''
    Parse the Korea Uni. MI data in a MATLAB formate that will be used in the next analysis

    The URL based fetching is a primitive code. So, please make sure not to interrupt it.
    Also, if you interrupt the process for any reason, remove the last downloaded subjects data.
    This is because, it's highly likely that the downloaded file for that subject will be corrupt.

    In spite of all this, make sure that you have close to 100GB free disk space
    and 70GB network bandwidth to properly download and save the MI data.

    Parameters
    ----------
    datasetPath : str
        Path to the BCI IV2a original dataset in gdf formate.
    savePath : str
        Path on where to save the epoched EEG data in a mat format.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list :
        channels to select from the data.
    downsampleFactor : int / None, optional
        down-sampling factor to use. The default is 4.
    verbos : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.
    The dataset will be saved at savePath.

    '''
    # Base url for fetching any data that is not present!
    fetchUrlBase = 'ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/'
    subjects = list(range(54))
    subAll = [subjects, subjects]
    subL = ['s', 'se']  # s: session 1, se: session 2 (session evaluation)

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSubs, subs in enumerate(subAll):
        for iSub, sub in enumerate(subs):
            print('Processing subject No.: ' + subL[iSubs] + str(iSub + 1).zfill(3))
            if not os.path.exists(os.path.join(savePath, subL[iSubs] + str(iSub + 1).zfill(3) + '.mat')):
                fileUrl = fetchUrlBase + 'session' + str(iSubs + 1) + '/' + 's' + str(iSub + 1) + '/' + 'sess' + str(
                    iSubs + 1).zfill(2) + '_' + 'subj' + str(iSub + 1).zfill(2) + '_EEG_MI.mat'
                data = fetchAndParseKoreaFile(
                    os.path.join(datasetPath, 'session' + str(iSubs + 1), 's' + str(iSub + 1), 'EEG_MI.mat'),
                    fileUrl, epochWindow=epochWindow, chans=chans, downsampleFactor=downsampleFactor)

                savemat(os.path.join(savePath, subL[iSubs] + str(iSub + 1).zfill(3) + '.mat'), data)


def matToPython(datasetPath, savePath, isFiltered=False):
    '''
    Convert the mat data to eegdataset and save it!

    Parameters
    ----------
    datasetPath : str
        path to mat dataset
    savePath : str
        Path on where to save the epoched eeg data in a eegdataset format.
    isFiltered : bool
        Indicates if the mat data is in the chan*time*trials*FilterBand format.
        default: False

    Returns
    -------
    None.

    '''
    print('Creating python eegdataset with raw data ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # load all the mat files
    data = [];
    for root, dirs, files in os.walk(datasetPath):
        files = sorted(files)
        for f in files:
            parD = {}
            parD['fileName'] = f
            parD['data'] = {}
            d = loadmat(os.path.join(root, f),
                        verify_compressed_data_integrity=False)
            if isFiltered:
                parD['data']['eeg'] = np.transpose(d['x'], (2, 0, 1, 3)).astype('float32')
            else:
                parD['data']['eeg'] = np.transpose(d['x'], (2, 0, 1)).astype('float32')

            parD['data']['labels'] = d['y']
            data.append(parD)

    # Start writing the files:
    # save the data in the eegdataloader format.
    # 1 file per sample in a dictionary formate with following fields:
    # id: unique key in 00001 formate
    # data: a 2 dimensional data matrix in chan*time formate
    # label: class of the data
    # Create another separate file to store the epoch info data.
    # This will contain all the intricate data division information.
    # There will be one entry for every data file and will be stored as a 2D array and in csv file.
    # The column names are as follows:
    # id, label -> these should always be present.
    # Optional fields -> subject, session. -> they will be used in data sorting.

    id = 0
    dataLabels = [['id', 'relativeFilePath', 'label', 'subject', 'session']]  # header row
    for i, d in enumerate(data):

        sub = int(d['fileName'][-7:-4])  # subject of the data
        sub = str(sub).zfill(3)

        if d['fileName'][1] == 'e':
            session = 1;
        elif d['fileName'][1] == '-':
            session = int(d['fileName'][2:4])
        else:
            session = 0;

        if len(d['data']['labels']) == 1:
            d['data']['labels'] = np.transpose(d['data']['labels'])
        # for j, label in enumerate(d['data']['labels']):
        #     print(j,label)
        for j, label in enumerate(d['data']['labels']):
            lab = label[0]
            # get the data
            if isFiltered:
                x = {'id': id, 'data': d['data']['eeg'][j, :, :, :], 'label': lab}
            else:
                x = {'id': id, 'data': d['data']['eeg'][j, :, :], 'label': lab}

            # dump it in the folder
            with open(os.path.join(savePath, str(id).zfill(5) + '.dat'), 'wb') as fp:
                pickle.dump(x, fp)

            # add in data label file
            dataLabels.append([id, str(id).zfill(5) + '.dat', lab, sub, session])

            # increment id
            id += 1
    # Write the dataLabels file as csv
    with open(os.path.join(savePath, "dataLabels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataLabels)

    # write miscellaneous data info as csv file
    dataInfo = [['fs', 250], ['chanName', 'Check Original File']]
    with open(os.path.join(savePath, "dataInfo.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataInfo)


def pythonToMultiviewPython(datasetPath, savePath,
                            filterTransform={'filterBank': {
                                'filtBank': [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32],
                                             [32, 36], [36, 40]], 'fs': 250, 'filtType': 'filter'}}):
    '''
    Convert the raw EEG data into its multi-view representation using a filter-bank
    specified with filterTransform.

    Parameters
    ----------
    datasetPath : str
        path to mat dataset
    savePath : str
        Path on where to save the epoched eeg data in a eegdataset format.
    filterTransform: dict
        filterTransform is a transform argument used to define the filter-bank.
        Please use the default settings unless you want to experiment with the filters.
        default: {'filterBank':{'filtBank':[[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]],'fs':250}}

    Returns
    -------
    None.
    Creates a new dataset and stores it in a savePath folder.

    '''
    # 此处加入了 filter trans form
    trasnformAndSave(datasetPath, savePath, transform=filterTransform)


def trasnformAndSave(datasetPath, savePath, transform=None):
    '''
    Apply a data transform and save the result as a new eegdataset

    Parameters
    ----------
    atasetPath : str
        path to mat dataset
    savePath : str
        Path on where to save the epoched eeg data in a eegdataset format.
    filterTransform: dict
        A transform to be applied on the data.

    Returns
    -------
    None.

    '''

    if transform is None:
        return -1

    # Data options:
    config = {}
    config['preloadData'] = False  # process One by one
    config['transformArguments'] = transform
    config['inDataPath'] = datasetPath
    config['inLabelPath'] = os.path.join(config['inDataPath'], 'dataLabels.csv')

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Outputs will be saved in folder : ' + savePath)

    # Check and compose transforms
    # FB ???
    if len(config['transformArguments']) > 1:
        transform = transforms.Compose(
            [transforms.__dict__[key](**value) for key, value in config['transformArguments'].items()])
    else:
        # 调用 transforms包里面的 filterBank 函数，创建一个filterbank的实例，名字是transform
        transform = transforms.__dict__[list(config['transformArguments'].keys())[0]](
            **config['transformArguments'][list(config['transformArguments'].keys())[0]])

    # Load the data
    # torch的load data
    data = eegDataset(dataPath=config['inDataPath'], dataLabelsPath=config['inLabelPath'],
                      preloadData=config['preloadData'], transform=transform)

    # Write the transform applied data
    dLen = len(data)
    perDone = 0

    for i, d in enumerate(data):
        with open(os.path.join(savePath, data.labels[i][1]), 'wb') as fp:  # 1-> realtive-path
            pickle.dump(d, fp)
        if i / dLen * 100 > perDone:
            print(str(perDone) + '% Completed')
            perDone += 1

    # Copy the labels and config files
    copyfile(config['inLabelPath'], os.path.join(savePath, 'dataLabels.csv'))
    copyfile(os.path.join(config['inDataPath'], "dataInfo.csv"), os.path.join(savePath, "dataInfo.csv"))

    # Store the applied transform in the transform . csv file
    with open(os.path.join(config['inDataPath'], "transform.csv"), 'w') as f:
        for key in config['transformArguments'].keys():
            f.write("%s,%s\n" % (key, config['transformArguments'][key]))


def fetchData(dataFolder, datasetId=0):
    '''
    Check if the rawMat, rawPython, and multiviewPython data exists
    if not, then create the above data

    Parameters
    ----------
    dataFolder : str
        The path to the parent dataFolder.
        example: '/home/FBCNetToolbox/data/korea/'
    datasetId : int
        id of the dataset:
            0 : bci42a data (default)
			1 : korea data

    Returns
    -------
    None.

    '''
    print('fetch ssettins: ', dataFolder, datasetId)
    oDataFolder = 'originalData'
    rawMatFolder = 'rawMat'
    rawPythonFolder = 'rawPython'
    multiviewPythonFolder = 'multiviewPython'

    # check that all original data exists
    if not os.path.exists(os.path.join(dataFolder, oDataFolder)):
        if datasetId == 0:
            raise ValueError('The BCI-IV-2a original dataset doesn\'t exist at path: ' +
                             os.path.join(dataFolder, oDataFolder) +
                             ' Please download and copy the extracted dataset at the above path ' +
                             ' More details about how to download this data can be found in the instructions.txt file')
        elif datasetId == 1:
            print('The Korea dataset doesn\'t exist at path: ' +
                  os.path.join(dataFolder, oDataFolder) +
                  ' So it will be automatically downloaded over FTP server ' +
                  'Please make sure that you have ~60GB Internet bandwidth and 80 GB space ' +
                  'the data size is ~60GB so its going to take a while ' +
                  'Meanwhile you can take a nap!')
        elif datasetId == 2:
            raise ValueError('The BCI-IV-2b original dataset doesn\'t exist at path: ' +
                             os.path.join(dataFolder, oDataFolder) +
                             ' Please download and copy the extracted dataset at the above path ' +
                             ' More details about how to download this data can be found in the instructions.txt file')
    else:
        oDataLen = len([name for name in os.listdir(os.path.join(dataFolder, oDataFolder))
                        if os.path.isfile(os.path.join(dataFolder, oDataFolder, name))])
        if datasetId == 0 and oDataLen < 36:
            raise ValueError('The BCI-IV-2a dataset at path: ' +
                             os.path.join(dataFolder, oDataFolder) +
                             ' is not complete. Please download and copy the extracted dataset at the above path ' +
                             'The dataset should contain 36 files (18 .mat + 18 .gdf)'
                             ' More details about how to download this data can be found in the instructions.txt file')
        elif datasetId == 1 and oDataLen < 108:
            print('The Korea dataset at path: ' +
                  os.path.join(dataFolder, oDataFolder) +
                  ' is incomplete. So it will be automatically downloaded over FTP server' +
                  ' Please make sure that you have ~60GB Internet bandwidth and 80 GB space' +
                  ' the data size is ~60GB so its going to take a while' +
                  ' Meanwhile you can take a nap!')
            parseKoreaDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))
        elif datasetId == 2 and oDataLen < 90:
            raise ValueError('The BCI-IV-2b dataset at path: ' +
                             os.path.join(dataFolder, oDataFolder) +
                             ' is not complete. Please download and copy the extracted dataset at the above path ' +
                             'The dataset should contain 36 files (18 .mat + 18 .gdf)'
                             ' More details about how to download this data can be found in the instructions.txt file')

    # Check if the processed mat data exists:
    if not os.path.exists(os.path.join(dataFolder, rawMatFolder)):
        print('Appears that the raw data exists but its not parsed yet. Starting the data parsing ')
        if datasetId == 0:
            parseBci42aDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))
        elif datasetId == 1:
            parseKoreaDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))
        if datasetId == 2:
            parseBci42bDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))

    # Check if the processed python data exists:
    if not os.path.exists(os.path.join(dataFolder, rawPythonFolder, 'dataLabels.csv')):
        print(
            'Appears that the parsed mat data exists but its not converted to eegdataset yet. Starting this processing')
        matToPython(os.path.join(dataFolder, rawMatFolder), os.path.join(dataFolder, rawPythonFolder))

    # Check if the multi-view python data exists:
    if not os.path.exists(os.path.join(dataFolder, multiviewPythonFolder, 'dataLabels.csv')):
        print(
            'Appears that the raw eegdataset data exists but its not converted to multi-view eegdataset yet. Starting this processing')
        pythonToMultiviewPython(os.path.join(dataFolder, rawPythonFolder),
                                os.path.join(dataFolder, multiviewPythonFolder))

    print('All the data you need is present! ')
    return 1

# parseBci42aDataset('/home/ravi/FBCNetToolbox/data/bci42a/originalData','/home/ravi/FBCNetToolbox/data/bci42a/rawMat')
# matToPython('/home/ravi/FBCNetToolbox/data/bci42a/rawMat','/home/ravi/FBCNetToolbox/data/bci42a/rawPython')
# pythonToMultiviewPython('/home/ravi/FBCNetToolbox/data/bci42a/rawPython','/home/ravi/FBCNetToolbox/data/bci42a/multiviewPython')
# fetchData('/home/ravi/FBCNetToolbox/data/korea/', 1)

# d = fetchAndParseKoreaFile('/home/ravi/FBCNetToolbox/data/korea/originalData/session1/s1/EEG_MI.mat', 
#                        'ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/session1/s1/sess01_subj01_EEG_MI.mat')

# parseKoreaDataset('/home/ravi/FBCNetToolbox/data/korea/originalData','/home/ravi/FBCNetToolbox/data/korea/rawMat')
