# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:03:55 2021

@author: songhama
"""
import json, os, glob
from scipy import interpolate
from more_itertools.recipes import grouper
import pandas as pd
import argparse

join = os.path.join
cur_d = os.path.dirname(__file__)

class DataCleaner:
    '''
    recover missing frames and incorrect detections
    
    Input:
        - filename: filename of the pose json. ex) PoseEstimation/output_fast/alphapose-results-fast-14_Trim.json
    '''
    def __init__(self, filename, out_folder):
        self.flabel = filename.split('/')[-1].split('.')[0]
        with open(filename, 'r') as of:
            self.data = json.load(of)
        self.tcks = []
        self.x = []
        self.ys = [[] for i in range(34)]
        self.newdata = []
        self.missing= 0
        self.wrong = 0
        self.out_folder = out_folder
        
    def filter_data(self):
        prev = None
        missing = 0
        prevwrong = 0
        
        cur_idx = int(self.data[0]['image_id'].split('.')[0]) - 1
        # cur_idx should be one less than frame_idx
        for i,d in enumerate(self.data):
            frame_idx = int(d['image_id'].split('.')[0])
            if frame_idx == cur_idx: 
                # skip same frame
                continue
            
            #missing frames
            if (frame_idx - cur_idx) > 1:
                self.missing += (frame_idx - cur_idx -1)
                cur_idx += (frame_idx - cur_idx -1)
                missing += 1
                
            poselist = list(grouper(d['keypoints'], 3))
            pose = pd.DataFrame(poselist)
            
            cur_idx += 1
            if i == 0:
                prev = pose
            else:
                diff = pose - prev
                #misdetection
                absdiff = max(diff.max()[:2].max(), -1* diff.min()[:2].min())
                if absdiff > 50:
                    if prevwrong:
                        prev=pose
                    else:
                        self.wrong += 1
                    prevwrong += 1
                    continue
            
            self.x.append(frame_idx)
            y_idx = 0
            for row in poselist:
                for value in row[:2]:
                    self.ys[y_idx].append(value)
                    y_idx += 1
            prev = pose
            missing = 0
            prevwrong = 0
            
    def interpolate(self):
        for y in self.ys:
            tck = interpolate.splrep(self.x, y)
            self.tcks.append(tck)
            
    def combine_keypoints(self, idx):
        keypoints = []
        for coor_list in self.ys:
            keypoints.append(coor_list[idx])
        return keypoints
    
    def recover_frame(self, frame_idx):
        keypoints = []
        cnt = -1
        for i in range(0,len(self.x),100):
            if self.x[i] > frame_idx:
                cnt = max(0, cnt)
                break
            cnt+=1
        for tck in self.tcks:
            kp = interpolate.splev(frame_idx, tck)
            keypoints.append(float(kp))
        return keypoints
    
    def fill_missing(self):
        frame_idx = 0
        for i,target_idx in enumerate(self.x):
            tmpdata = dict()
            if frame_idx == target_idx:
                tmpdata['image_id'] = frame_idx
                tmpdata['keypoints'] = self.combine_keypoints(i)
                self.newdata.append(tmpdata)
                frame_idx += 1
            
            #recover missing frames
            elif frame_idx < target_idx:
                while frame_idx <= target_idx:
                    tmpdata = dict()
                    tmpdata['image_id'] = frame_idx
                    tmpdata['keypoints'] = self.recover_frame(frame_idx)
                    self.newdata.append(tmpdata)
                    frame_idx += 1
            else:
                print("WRONG", frame_idx, target_idx)
                return
            
    def save(self):
        newfilename = join(self.out_folder, f"{self.flabel}.json")
        with open(newfilename, 'w') as of:
            json.dump(self.newdata, of)
        print("saved output")
        
            
    def run(self):
        print('filtering...')
        self.filter_data()
        print('interpolating...')
        self.interpolate()
        print('recovering...')
        self.fill_missing()
        print(f'done / {self.missing} missing frames, {self.wrong} misdetections fixed!')
        self.save()

def clean_all_files(args):
    files = glob.glob(join(args.input_folder, '*.json'))
    print(files)
    return
    for filename in files:
        dc = DataCleaner(filename, args.output_folder)
        dc.run()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "-i",
            "--input_folder",
            required=True,
            help="folder path of the input files. e.g. data/pose/",
            type=str
        )

    parser.add_argument(
            "-o",
            "--output_folder",
            default=join(cur_d, "fixed_poses"),
            help="folder path of the output",
            type=str
        )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    clean_all_files(parse_args())