# import k4a
import numpy as np
import torch
import clip
import os
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
from PIL import Image
from tqdm import tqdm

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
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    ax.imshow(img)

class ScenePerception():

    def __init__(self, sam_checkpoint_type, sam_checkpoint_path):

        # self.objects = instances # ["a hammer", "a pink cup", "a green cup", "a nail", "a pot", "a robot", "a table", "background"]

        self.sam = sam_model_registry[sam_checkpoint_type](checkpoint=sam_checkpoint_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sam.to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
                            self.sam
                        )

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    
    def get_masked_image(self, image, mask):
        mask = ~mask["segmentation"]
        masked_image = image.copy()
        masked_image[mask] = 1
        return masked_image

    def get_sam_masks(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        return masks

    def get_clip_probabilities(self, image, masks, instance_descriptions):

        all_probs = []

        for itr, mask in tqdm(enumerate(masks), total=len(masks)):
            masked_image = self.get_masked_image(image, mask)
            # get clip features
            pil_image = Image.fromarray(masked_image)
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize(instance_descriptions).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                logits_per_image, logits_per_text = self.model(image_input, text_input)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            # print(f"Mask {itr} - {probs}")
            all_probs.append(probs[0])
    
        all_probs = np.array(all_probs)

        return all_probs

    def get_the_object_masks(self, image, masks, all_probs, instance_descriptions):

        all_masks = []

        for ind, prompt in tqdm(enumerate(instance_descriptions), total=len(instance_descriptions)):
            # print(f"Prompt: {prompt}, {ind} - {all_probs[:, ind]}")
            sorted_indices = np.argsort(all_probs[:, ind])[::-1]

            mask = masks[sorted_indices[0]]
            masked_image = self.get_masked_image(image, mask)

            all_masks.append(mask)
        
        return all_masks

    def get_masks_from_dict(self, image, instance_dict):

        instance_keys = list(instance_dict.keys())
        instance_descriptions = list(instance_dict.values())

        print(instance_keys, instance_descriptions)

        sam_masks = self.get_sam_masks(image)

        # plt.figure(figsize=(20,20))
        # plt.imshow(image)
        # show_anns(sam_masks)
        # plt.axis('off')
        # plt.show() 

        # assert False

        clip_probs = self.get_clip_probabilities(image, sam_masks, instance_descriptions)
        object_masks = self.get_the_object_masks(image, sam_masks, clip_probs, instance_descriptions)

        mask_dict = {}
        for key, mask in zip(instance_keys, object_masks):
            mask_dict[key] = mask

        return mask_dict

    def debug_mask_dict_to_image(self, image, mask_dict):

        for key in list(mask_dict.keys()):
            masked_image = self.get_masked_image(image, mask_dict[key])
            plt.imshow(masked_image)
            plt.savefig(f"all_masks2/mask_{key}.png")


if __name__=="__main__":

    sam_checkpoint_type = "default"
    sam_checkpoint_path = "/home/rl2-ws1/Downloads/sam_vit_h_4b8939.pth"
    test_image_all = "./workspace2.jpg" # "./real_robot_setup.png"
    test_image_all = cv2.imread(test_image_all)
    test_image_all = cv2.resize(test_image_all, (test_image_all.shape[1]//5, test_image_all.shape[0]//5))
    test_image_all = cv2.cvtColor(test_image_all, cv2.COLOR_BGR2RGB)

    perception_module = ScenePerception(sam_checkpoint_type, sam_checkpoint_path)

    instance_dict = {
        "hammer": "a hammer",
        "cup2": "a green cup",
        "cup1": "a pink cup",
        "nail": "a yellow nail",
        "pot" : "a kitchen pot",
        "robot": "a robot",
        "table": "a white table",
        "background": "a background"
    }

    mask_dict = perception_module.get_masks_from_dict(test_image_all, instance_dict)
    perception_module.debug_mask_dict_to_image(test_image_all, mask_dict)


