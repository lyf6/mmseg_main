import argparse
import os
from slicedata.slice.steamdata import SteamData
from slicedata.utils.combine import Combine
from slicedata.utils.io import save_img

import torch
from mmseg.apis import init_model
from batch_infer.infer import semantic_segment_infer
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pycocotools.mask as maskUtils
import numpy as np
import cv2
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
    parser.add_argument(
        "--config", dest="config", help="The config file.", default='configs/segformer/segformer_mit-b5_8xb1-160k_corn-1024x1024.py', type=str)

    parser.add_argument(
        '--checkpoint_file',
        dest='checkpoint_file',
        help='The path of model for evaluation.',
        type=str,
        default='work_dirs/segformer_mit-b5_8xb1-160k_corn-1024x1024/iter_40000.pth')
    
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Number of workers for data loader.',
        type=int,
        default=0)
    
    parser.add_argument(
        '--device',
        dest='device',
        help='Device place to be set, which can be gpu, xpu, npu, or cpu.',
        default='cuda',
        choices=['cpu', 'cuda', 'xpu', 'npu'],
        type=str)
    
    parser.add_argument(
        '--slice_size',
        dest='slice_size',
        nargs=2,
        help='The size of sliding window, the first is width and the second is height.',
        type=int,
        default=[1400, 840])
    
    parser.add_argument(
        '--overlap',
        dest='overlap',
        nargs=2,
        help='The overlap of sliding window, the first is width and the second is height.',
        type=float,
        default=[0.25, 0.25])

    parser.add_argument(
        '--image_dir',
        dest='image_dir',
        help='The dir path of images for evaluation.',
        type=str,
        default='test_data/test')
    
    parser.add_argument(
        '--image_list',
        dest='image_list',
        help='The names of images for evaluation.',
        type=str,
        default=None)
    
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The dir of saving mask',
        type=str,
        default='test_data/result')
    
    # parser.add_argument(
    #     '--batch_size',
    #     dest='batch_size',
    #     help='batch_size for data loader.',
    #     type=int,
    #     default=32)    
    parser.add_argument(
        '--combine_method',
        help='method to save combine mask, average_logit or voting_mask, only voting supported now',
        type=str,
        default='average_logit')    
    return parser.parse_args()

def load_semantic_model(config_file, checkpoint_file):
    model = init_model(config_file, checkpoint_file)
    return model



def load_segment_everything(model_type='vit_h',
                             points_per_side=24,
                             pred_iou_thresh=0.86,
                             stability_score_thresh=0.92,
                             crop_n_layers=1,
                             crop_n_points_downscale_factor=2,
                             min_mask_region_area=100,
                             output_mode='coco_rle',
                             device='cuda'
                             ):
    model_map = {
        'vit_h':'vit_segment_anything/vit_h.pth',
        'vit_b':'vit_segment_anything/vit_b.pth',
        'vit_l':'vit_segment_anything/vit_l.pth'
    }
    sam = sam_model_registry[model_type](checkpoint=model_map[model_type]).to(device)
    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,  # Requires open-cv to run post-processing
        output_mode=output_mode,
    )
    return mask_branch_model

def remove_maoci(mask, kernel=9, iterations=2):
    kernel = np.ones((kernel, kernel), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel, iterations=iterations)

    return opening

def thin(mask, kernel=7, iterations=1):
    kernel = np.ones((kernel, kernel), dtype=np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=iterations)
    return erosion

def remove_hole(mask, kernel=5, iterations=2):
    kernel = np.ones((kernel, kernel), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=iterations)

    return opening

# def get_all_segment()

def get_fine_mask(segment_everything, img, semantc_mask):
    anns = {'annotations': segment_everything.generate(img)}
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'])
    
    class_ids = torch.zeros_like(semantc_mask)
    index = torch.zeros_like(class_ids).bool()


    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool().to(index.device)
        valid_mask = valid_mask*(~index)
        index[valid_mask] = True
        # get the class ids of the valid pixels
        propose_classes_ids = semantc_mask[valid_mask]
        num_class_proposals = len(torch.unique(propose_classes_ids))
        if num_class_proposals == 1:
            class_ids[valid_mask] = propose_classes_ids[0]
            continue
        # top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        # when there is 30% pixel, it should be regared as true
        top_propose_class_ids = torch.bincount(propose_classes_ids.flatten())
        sum = top_propose_class_ids.sum()
        top = top_propose_class_ids[1:].topk(1)
        top_value = top.values
        if(top_value/sum>=0.2):
            top_1_propose_class_ids = top.indices + 1
        else:
            top_1_propose_class_ids = 0

        if(top_1_propose_class_ids>0): # 0 is background
            use_mask = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
            use_mask[valid_mask.cpu().numpy()] = 1
            opening = thin(use_mask)
            local_index = opening>0
            class_ids[local_index] = top_1_propose_class_ids.type(class_ids.dtype)
        else: 
            class_ids[valid_mask] = top_1_propose_class_ids
        
        del valid_mask
        del propose_classes_ids
        del num_class_proposals
        del top_1_propose_class_ids
    
    class_ids = class_ids.cpu().numpy().astype(np.uint8)


    class_ids = remove_maoci(class_ids)

    # class_ids = remove_hole(class_ids)
    
    # a = Image.fromarray(class_ids*255)
    # a.save('a.png')
    return class_ids

def combine_labelid(semantic_model, segment_everything, loader,  combine_tool, save_path, eval_dataset):
    with torch.no_grad():
        for iter, data in enumerate(loader):
            mask, _ = semantic_segment_infer(semantic_model, data['img'][0].numpy())
            mask = get_fine_mask(segment_everything, data['img'][0].numpy(), mask)
            data['mask'] = mask
            combine_tool.combine_mask(data)

    fine_mask = combine_tool.get_mask_bymask()*255
    save_img(fine_mask, save_path, in_ds = eval_dataset.large_data.data_info.get('GeoTransform_And_Projection', None))


def combine_logit(semantic_model, segment_everything, loader,  combine_tool, save_path, eval_dataset, device):
    with torch.no_grad():
        for iter, data in enumerate(loader):
            _, prob = semantic_segment_infer(semantic_model, data['img'][0].numpy())
            # mask = get_fine_mask(segment_everything, data['img'][0].numpy(), mask)
            data['logit'] = prob.cpu().numpy()
            combine_tool.combine_logit(data)
        
        combine_mask = combine_tool.get_mask_bylogit()
        # a = Image.fromarray(combine_mask*255)
        # a.save('large_ori.png')
        # fine_mask = np.zeros_like(combine_mask).astype(np.uint8)
        for iter, data in enumerate(loader):
            start_h, start_w = data['start_loc'] 
            valid_height, valid_width = data['valid']
            semantic_mask = torch.from_numpy(combine_mask[start_h:start_h+valid_height, start_w:start_w+valid_width]).to(device)
            mask = get_fine_mask(segment_everything, data['img'][0].numpy(), semantic_mask)
            data['mask'] = mask
            combine_tool.combine_mask(data)
    
        fine_mask = combine_tool.get_mask_bymask()*255
        # a = Image.fromarray(fine_mask)
        # a.save('large_fine.png')
        save_img(fine_mask, save_path, in_ds = eval_dataset.large_data.data_info.get('GeoTransform_And_Projection', None))


    


def steam_predict(semantic_model,
            segment_everything,
            slice_height=1024, 
            slice_width=1024, 
            overlap_height_ratio=0.2, 
            overlap_width_ratio=0.2,
            image_dir=None,
            image_list=None,
            save_dir='output',
            num_workers=0,
            combine_method='voting_mask',
            device='cuda'):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
       
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """

    # semantic_model.eval()
    num_classes = semantic_model.num_classes
    for img_name in image_list:
        
        img_path = os.path.join(image_dir, img_name)
        save_path = os.path.join(save_dir, img_name)
        
        eval_dataset = SteamData(img_path,slice_height, slice_width, overlap_height_ratio, overlap_width_ratio)
        combine_tool = Combine(eval_dataset.large_data.data_info, num_classes, combine_method)
        # batch_sampler = torch.utils.data.SequentialSampler(eval_dataset)
        loader = torch.utils.data.DataLoader(
            eval_dataset,
            # batch_sampler=batch_sampler,
            num_workers=num_workers,
            shuffle=False)


        # with torch.no_grad():
        #     for iter, data in enumerate(loader):
        #         mask = semantic_segment_infer(semantic_model, data['img'][0].numpy())
        #         mask = get_fine_mask(segment_everything, data['img'][0].numpy(), mask)
        #         data['mask'] = mask
        #         combine_tool.combine(data)
        
        # fine_mask = combine_tool.get_mask()*255
        # save_img(fine_mask, save_path, in_ds = eval_dataset.large_data.data_info.get('GeoTransform_And_Projection', None))
        if(combine_method=='voting_mask'):
            combine_labelid(semantic_model, segment_everything, loader,  combine_tool, save_path, eval_dataset)
        else:
            combine_logit(semantic_model, segment_everything, loader,  combine_tool, save_path, eval_dataset, device=device)
        
def slice_infer():
    args = parse_args()
    device = torch.device('cuda')
    semantic_model = load_semantic_model(args.config, args.checkpoint_file).to(device)
    segment_everything = load_segment_everything(device=device)
    slice_width, slice_height= args.slice_size
    overlap_width_ratio, overlap_height_ratio= args.overlap
    image_dir=args.image_dir
    image_list=[]
    save_dir=args.save_dir
    combine_method=args.combine_method
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    num_workers = args.num_workers
    if(args.image_list is not None):
        with open(args.image_list, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                image_list.append(line)
    else:
        image_list = os.listdir(image_dir)
    
    steam_predict(semantic_model,
                  segment_everything,
                  slice_height=slice_height, 
                  slice_width=slice_width, 
                  num_workers=num_workers,
                  overlap_height_ratio=overlap_height_ratio,
                  overlap_width_ratio=overlap_width_ratio,
                  image_dir=image_dir,
                  image_list=image_list,
                  save_dir=save_dir,
                  combine_method=combine_method,
                  device=device)

if __name__ == '__main__':
    slice_infer()
