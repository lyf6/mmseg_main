from mmseg.apis import inference_model, init_model
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import mmcv
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt
import cv2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



id2label={
"id2label": {
    "0": "bg",
    "1": "corn"
  },
}

def semantic_segment_anything_inference(
        img_path:str,
        config_file:str,
        checkpoint_file:str,
        mask_branch_model:SamAutomaticMaskGenerator,
        ori_save_path:str='ori.png',
        fine_save_path:str='fine.png',
):
    
    img = mmcv.imread(img_path)
    anns = {'annotations': mask_branch_model.generate(img)}
    # image = cv.imread('images/dog.jpg')
    # image = cv.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(20,20))
    # plt.imshow(img)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show() 
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    result = inference_model(model, img)
    class_ids = result.pred_sem_seg.data.squeeze()
    semantc_mask = class_ids.clone()
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
    class_names = []
    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        # get the class ids of the valid pixels
        propose_classes_ids = class_ids[valid_mask]
        num_class_proposals = len(torch.unique(propose_classes_ids))
        if num_class_proposals == 1:
            semantc_mask[valid_mask] = propose_classes_ids[0]
            ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            class_names.append(ann['class_name'])
            # bitmasks.append(maskUtils.decode(ann['segmentation']))
            continue
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]

        semantc_mask[valid_mask] = top_1_propose_class_ids
        ann['class_name'] = top_1_propose_class_names[0]
        ann['class_proposals'] = top_1_propose_class_names[0]
        class_names.append(ann['class_name'])
        # bitmasks.append(maskUtils.decode(ann['segmentation']))

        del valid_mask
        del propose_classes_ids
        del num_class_proposals
        del top_1_propose_class_ids
        del top_1_propose_class_names
    

    # sematic_class_in_img = torch.unique(semantc_mask)
    # semantic_bitmasks, semantic_class_names = [], []

    # semantic prediction
    # anns['semantic_mask'] = {}
    # for i in range(len(sematic_class_in_img)):
    #     class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]
    #     class_mask = semantc_mask == sematic_class_in_img[i]
    #     class_mask = class_mask.cpu().numpy().astype(np.uint8)
    #     semantic_class_names.append(class_name)
    #     semantic_bitmasks.append(class_mask)
    #     anns['semantic_mask'][str(sematic_class_in_img[i].item())] = maskUtils.encode(np.array((semantc_mask == sematic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
    #     anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'] = anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'].decode('utf-8')
    # 手动清理不再需要的变量
    class_ids = class_ids.cpu().numpy().astype(np.uint8)*255
    semantc_mask = semantc_mask.cpu().numpy().astype(np.uint8)*255
    class_ids=Image.fromarray(class_ids)
    semantc_mask=Image.fromarray(semantc_mask)
    class_ids.save('origin.png')
    semantc_mask.save('fine.png')
    del img
    del anns
    del class_ids
    del semantc_mask
    # del bitmasks
    del class_names
 
    # return semantc_mask   





config_file = 'configs/segformer/segformer_mit-b5_8xb1-160k_corn-1024x1024.py'
checkpoint_file = 'work_dirs/segformer_mit-b5_8xb1-160k_corn-1024x1024/iter_40000.pth'
sam_check_point = 'vit_segment_anything/vit_h.pth'
img_path='test_data/test/tif1.png'
target_path='test.png'


sam = sam_model_registry["vit_h"](checkpoint=sam_check_point).to(0)
mask_branch_model = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    # Foggy driving (zero-shot evaluate) is more challenging than other dataset, so we use a larger points_per_side
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
    output_mode='coco_rle',
)

with torch.no_grad():
    semantic_segment_anything_inference(img_path, config_file, checkpoint_file, mask_branch_model)
