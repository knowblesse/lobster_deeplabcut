import numpy as np
from matplotlib import pyplot as plt
import deeplabcut as dlc

task_name = 'Test'
experimenter = 'Knowblesse'
video=[]
path_config_file = dlc.create_new_project(task_name, experimenter, video)

dlc.add_new_videos(path_config_file,['./Test-Knowblesse-2020-04-08/videos/Lobster_Training-190725-161613_19DEC1-200210-111439_Vid1.avi'])
dlc.add_new_videos(path_config_file,['./Test-Knowblesse-2020-04-08/videos/Lobster_Training-190725-161613_19DEC5-200204-104703_Vid1.avi'])

path_config_file = '/home/knowblesse/VCF/lobster_deeplabcut/Test-Knowblesse-2020-04-08/config.yaml'
dlc.extract_frames(path_config_file)
dlc.label_frames(path_config_file)

#
videofile_path = ['Lobster_Training-190725-161613_19DEC3-200205-104339_Vid1_novel.avi']

dlc.analyze_videos(path_config_file,videofile_path, videotype='.avi')
dlc.create_labeled_video(path_config_file,videofile_path,draw_skeleton=True)