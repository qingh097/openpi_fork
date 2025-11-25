import csv
import os
import torch
import h5py
import imageio
import numpy as np
IMAGE_SKIP_FRAMES = 45

def curate_init_state_index(root_folder):
    init_state = {}
    init_state_path = os.path.join(root_folder, 'init_state_path.csv')
    with open(init_state_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            task, seed, index = row
            seed_init_state = torch.load(os.path.join(root_folder, f'seed_{seed}', f'{task}.pruned_init'))
            cur_task_init_state = init_state.get(task, [])
            cur_task_init_state.append(seed_init_state[int(index)])
            init_state[task] = cur_task_init_state
    return init_state

def curate_goal_images(root_folder):
    agentview_goal_images = {}
    wrist_goal_images = {}
    index_path = os.path.join(root_folder, 'init_state_path.csv')
    with open(index_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            task, seed, index = row
            trajectory_path = os.path.join(root_folder, f'seed_{seed}', f'{task}.hdf5')
            trajectory_data = h5py.File(trajectory_path, 'r')['data']
            all_agentview_image = trajectory_data[f'demo_{index}/obs/agentview_rgb'][:][:,::-1,::-1]
            all_wrist_image = trajectory_data[f'demo_{index}/obs/eye_in_hand_rgb'][:][:,::-1,::-1]
            
            agentview_image = all_agentview_image[::IMAGE_SKIP_FRAMES][1:] #skip the first image
            wrist_image = all_wrist_image[::IMAGE_SKIP_FRAMES][1:] #skip the first image
            #always append the last image
            agentview_image = np.concatenate([agentview_image, all_agentview_image[-1:]], axis=0)
            wrist_image = np.concatenate([wrist_image, all_wrist_image[-1:]], axis=0)
            
            cur_task_agentview_images = agentview_goal_images.get(task, [])
            cur_task_wrist_images = wrist_goal_images.get(task, [])
            cur_task_agentview_images.append(agentview_image)
            cur_task_wrist_images.append(wrist_image)
            agentview_goal_images[task] = cur_task_agentview_images
            wrist_goal_images[task] = cur_task_wrist_images
    return agentview_goal_images, wrist_goal_images
            

root_folder = '/viscam/projects/dexs2r/libero_init/libero_object_unseen'
agentview_goal_images_all, wrist_goal_images_all = curate_goal_images(root_folder)

if True:
    #visualize the goal images by saving to video
    for task, agentview_goal_images in agentview_goal_images_all.items():
        for i in range(len(agentview_goal_images)):
            agentview_image = agentview_goal_images[i] #T,H,W,3
            wrist_image = wrist_goal_images_all[task][i] #T,H,W,3
            #save the image to a video
            video_path = os.path.join(root_folder,f'{task}_agentview_goal_images_{i}.mp4')
            imageio.mimsave(video_path, agentview_image)
            video_path = os.path.join(root_folder,f'{task}_wrist_goal_images_{i}.mp4')
            imageio.mimsave(video_path, wrist_image)

for task, agentview_goal_images in agentview_goal_images_all.items():
    torch.save(agentview_goal_images, os.path.join(root_folder,f'{task}_agentview_goal_images.pt'))
for task, wrist_goal_images in wrist_goal_images_all.items():
    torch.save(wrist_goal_images, os.path.join(root_folder,f'{task}_wrist_goal_images.pt'))


# init_state = curate_init_state_index(root_folder)
# for task, init_state in init_state.items():
#     torch.save(init_state, os.path.join(root_folder,f'{task}_all.pruned_init'))