''' Script that selects first available GPU'''
import os
import GPUtil
devicesid=GPUtil.getFirstAvailable()
os.environ["CUDA_VISIBLE_DEVICES"] = str(devicesid[0])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
print('GPU %d was selected' %devicesid[0])
