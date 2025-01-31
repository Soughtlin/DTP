"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import re
from PIL import Image

import pandas as pd
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor

from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset


class LVUCLSDataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 history, num_frames, task, stride=10, split='train'):
        """
        vis_root (string): Root directory of videos (e.g. LVU/videos/)
        ann_root (string): directory to store the gt_dict file
        """
        self.vis_root = vis_root

        task_list = ['director', 'genre', 'relationship', 'scene', 'way_speaking', 'writer', 'year']
        assert task in task_list, f'Invalid task {task}, must be one of {task_list}'
        self.task = task

        self.gt_dict = {}
        for ann_path in ann_paths:
            self.gt_dict.update(json.load(open(ann_path)))

        self.fps = 10
        self.annotation = {}
        self.stride = stride
        for video_id in self.gt_dict:
            if task in self.gt_dict[video_id]:
                duration = self.gt_dict[video_id]['duration']
                video_length = self.gt_dict[video_id]['num_frames']
                label = self.gt_dict[video_id][task]
                label_after_process = text_processor(label)
                assert label == label_after_process, "{} not equal to {}".format(label, label_after_process)
                self.annotation[f'{video_id}_0'] = {'video_id': video_id, 'start': 0, 'label': label_after_process,
                                                    'duration': duration, 'video_length': video_length,
                                                    'answer': self.gt_dict[video_id][f'{task}_answer']}
                for start in range(self.stride, duration - history + 1, self.stride):
                    video_start_id = f'{video_id}_{start}'
                    self.annotation[video_start_id] = {'video_id': video_id, 'start': start,
                                                       'label': label_after_process, 'duration': duration,
                                                       'video_length': video_length,
                                                       'answer': self.gt_dict[video_id][f'{task}_answer']}

        self.data_list = list(self.annotation.keys())
        self.data_list.sort()

        self.history = history
        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        video_start_id = self.data_list[index]

        start_time = self.annotation[video_start_id]['start']
        end_time = min(self.annotation[video_start_id]['start'] + self.history - 1,
                       self.annotation[video_start_id]['duration'])

        start_frame_index = int(start_time * self.fps)
        end_frame_index = min(int(end_time * self.fps), self.annotation[video_start_id]['video_length'] - 1)
        selected_frame_index = np.rint(np.linspace(start_frame_index, end_frame_index, self.num_frames)).astype(
            int).tolist()
        # print(start_frame_index, end_frame_index, selected_frame_index, start_time, end_time)
        merge_list = []
        C_target = 4
        for frame_index in selected_frame_index:
            # frame = Image.open(os.path.join(self.vis_root, self.annotation[video_start_id]['video_id'], "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            # '/home/data2/lvu_video/frames'
            frame = Image.open(os.path.join('/lrh/VideoUnderstanding/datasets/MALMM_data/LVU_frames',
                                            self.annotation[video_start_id]['video_id'],
                                            "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)

            mask_path = os.path.join('/home/data1/MA-LMM/new_dataset/processed/lvu_mask/masks',
                                     self.annotation[video_start_id]['video_id'],
                                     "frame{:06d}.npy".format(frame_index + 1))
            if not os.path.exists(mask_path):
                mask = np.zeros((C_target, frame.shape[1], frame.shape[2]))
            else:
                mask = np.load(mask_path).astype(int) * 255

            if frame is None or mask is None:
                print('ERROR NONE ',
                      os.path.join(self.vis_root, ann['video'], "frame{:06d}.jpg".format(frame_index + 1)))

            C_mask, H_mask, W_mask = mask.shape

            # numpy version
            if C_mask > C_target:
                sampled_indices = np.random.permutation(C_mask)[:C_target]
                mask_tensor = mask[sampled_indices, :, :].copy()

            elif C_mask < C_target:
                sampled_indices = np.random.permutation(C_target) % C_mask
                mask_tensor = mask[sampled_indices, :, :].copy()
            else:
                mask_tensor = mask.copy()

            # merging
            mask_tensor = torch.from_numpy(mask_tensor).to(torch.float32)
            merge_frame = torch.cat([frame, mask_tensor], dim=0).numpy()
            merge_list.append(merge_frame)

            del mask

        # video = torch.stack(frame_list, dim=1)
        # video = self.vis_processor(video)

        merge_array = np.asarray(merge_list)
        video_and_mask = torch.from_numpy(merge_array).transpose(0, 1)

        # video_and_mask = torch.stack(merge_list, dim=1)
        video, masks = self.vis_processor(video_and_mask)  # C T H W

        text_input = self.text_processor(f'what is the {self.task} of the movie?')
        caption = self.text_processor(self.annotation[video_start_id]['label'])
        return {
            "image": video,
            "mask": masks,
            "text_input": text_input,
            "text_output": caption,
            "image_id": video_start_id,
            "question_id": video_start_id,
        }

    def __getitem__ori(self, index):
        video_start_id = self.data_list[index]

        start_time = self.annotation[video_start_id]['start']
        end_time = min(self.annotation[video_start_id]['start'] + self.history - 1,
                       self.annotation[video_start_id]['duration'])

        start_frame_index = int(start_time * self.fps)
        end_frame_index = min(int(end_time * self.fps), self.annotation[video_start_id]['video_length'] - 1)
        selected_frame_index = np.rint(np.linspace(start_frame_index, end_frame_index, self.num_frames)).astype(
            int).tolist()
        # print(start_frame_index, end_frame_index, selected_frame_index, start_time, end_time)
        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join('/lrh/VideoUnderstanding/datasets/MALMM_data/LVU_frames',
                                            self.annotation[video_start_id]['video_id'],
                                            "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        text_input = self.text_processor(f'what is the {self.task} of the movie?')
        caption = self.text_processor(self.annotation[video_start_id]['label'])
        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "image_id": video_start_id,
            "question_id": video_start_id,
        }

    def __len__(self):
        return len(self.data_list)


class LVUCLSEvalDataset(LVUCLSDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 history, num_frames, task, stride=10, split='val'):

        super().__init__(vis_processor, text_processor, vis_root, ann_paths,
                         history, num_frames, task, stride=stride, split=split)

    def __getitem__(self, index):
        video_start_id = self.data_list[index]

        start_time = self.annotation[video_start_id]['start']
        end_time = min(self.annotation[video_start_id]['start'] + self.history - 1,
                       self.annotation[video_start_id]['duration'])

        start_frame_index = int(start_time * self.fps)
        end_frame_index = min(int(end_time * self.fps), self.annotation[video_start_id]['video_length'] - 1)
        selected_frame_index = np.rint(np.linspace(start_frame_index, end_frame_index, self.num_frames)).astype(
            int).tolist()
        # print(start_frame_index, end_frame_index, selected_frame_index, start_time, end_time)
        merge_list = []
        C_target = 4
        for frame_index in selected_frame_index:
            # frame = Image.open(os.path.join(self.vis_root, self.annotation[video_start_id]['video_id'], "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            # /home/data2/lvu_video/
            frame = Image.open(os.path.join('/lrh/VideoUnderstanding/datasets/MALMM_data/LVU_frames',
                                            self.annotation[video_start_id]['video_id'],
                                            "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)

            mask_path = os.path.join('/home/data1/MA-LMM/new_dataset/processed/lvu_mask/masks',
                                     self.annotation[video_start_id]['video_id'],
                                     "frame{:06d}.npy".format(frame_index + 1))
            if not os.path.exists(mask_path):
                mask = np.zeros((C_target, frame.shape[1], frame.shape[2]))
            else:
                mask = np.load(mask_path).astype(int) * 255

            if frame is None or mask is None:
                print('ERROR NONE ',
                      os.path.join(self.vis_root, ann['video'], "frame{:06d}.jpg".format(frame_index + 1)))

            # if C_mask > C_target:
            #    sampled_indices = torch.randperm(C_mask)[:C_target]
            #    mask_tensor = mask[sampled_indices, :, :]
            # other mask version
            # _mask = mask[:C_target - 1, :, :]
            # other_mask =  torch.max(mask[C_target - 1 : , :, :],dim=0).values.unsqueeze(0)
            # mask_tensor = torch.cat([_mask, other_mask], dim=0)
            # elif C_mask < C_target:
            #    sampled_indices = torch.randperm(C_target)[:C_target - C_mask] % C_mask
            #    mask_duplicate = mask[sampled_indices, :, :]
            #    mask_tensor = torch.cat([mask, mask_duplicate], dim=0)
            # else:
            #    mask_tensor = mask

            C_mask, H_mask, W_mask = mask.shape

            # numpy version
            if C_mask > C_target:
                sampled_indices = np.random.permutation(C_mask)[:C_target]
                mask_tensor = mask[sampled_indices, :, :].copy()

            elif C_mask < C_target:
                sampled_indices = np.random.permutation(C_target) % C_mask
                mask_tensor = mask[sampled_indices, :, :].copy()
            else:
                mask_tensor = mask.copy()

            # merging
            mask_tensor = torch.from_numpy(mask_tensor).to(torch.float32)
            merge_frame = torch.cat([frame, mask_tensor], dim=0).numpy()
            merge_list.append(merge_frame)

            del mask

        # video = torch.stack(frame_list, dim=1)
        # video = self.vis_processor(video)

        merge_array = np.asarray(merge_list)
        video_and_mask = torch.from_numpy(merge_array).transpose(0, 1)

        # video_and_mask = torch.stack(merge_list, dim=1)
        video, masks = self.vis_processor(video_and_mask)  # C T H W

        text_input = self.text_processor(f'what is the {self.task} of the movie?')
        caption = self.text_processor(self.annotation[video_start_id]['label'])
        return {
            "image": video,
            "mask": masks,
            "text_input": text_input,
            "prompt": text_input,
            "text_output": caption,
            "image_id": video_start_id,
            "question_id": video_start_id,
        }

    def __getitem__ori(self, index):
        video_start_id = self.data_list[index]

        start_time = self.annotation[video_start_id]['start']
        end_time = min(self.annotation[video_start_id]['start'] + self.history - 1,
                       self.annotation[video_start_id]['duration'])

        start_frame_index = int(start_time * self.fps)
        end_frame_index = min(int(end_time * self.fps), self.annotation[video_start_id]['video_length'] - 1)
        selected_frame_index = np.rint(np.linspace(start_frame_index, end_frame_index, self.num_frames)).astype(
            int).tolist()
        # print(start_frame_index, end_frame_index, selected_frame_index, start_time, end_time)
        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join('/home/data2/lvu_video/frames', self.annotation[video_start_id]['video_id'],
                                            "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        text_input = self.text_processor(f'what is the {self.task} of the movie?')
        caption = self.text_processor(self.annotation[video_start_id]['label'])
        return {
            "image": video,
            "text_input": text_input,
            "prompt": text_input,
            "text_output": caption,
            "image_id": video_start_id,
            "question_id": video_start_id,
        }
