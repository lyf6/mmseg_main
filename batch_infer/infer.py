from mmseg.apis import inference_model, init_model
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import mmcv
import pycocotools.mask as maskUtils


def semantic_segment_infer(
        model:torch.nn.Module,
        img:np.array
):
    result = inference_model(model, img)
    semantc_mask = result.pred_sem_seg.data.squeeze()
    prob = torch.softmax(result.seg_logits.data, dim=0).permute(1,2,0)
    # semantc_mask = class_ids.clone()
    # anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
    # class_names = []
    # for ann in anns['annotations']:
    #     valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
    #     # get the class ids of the valid pixels
    #     propose_classes_ids = class_ids[valid_mask]
    #     num_class_proposals = len(torch.unique(propose_classes_ids))
    #     if num_class_proposals == 1:
    #         semantc_mask[valid_mask] = propose_classes_ids[0]
         
    #         # bitmasks.append(maskUtils.decode(ann['segmentation']))
    #         continue
    #     top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        
    #     semantc_mask[valid_mask] = top_1_propose_class_ids
        
    #     # bitmasks.append(maskUtils.decode(ann['segmentation']))

    #     del valid_mask
    #     del propose_classes_ids
    #     del num_class_proposals
    #     del top_1_propose_class_ids

#     semantc_mask = semantc_mask.cpu().numpy().astype(np.uint8)
    

    return semantc_mask, prob

