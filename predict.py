import glob
import os
import time
import json
import onnx
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import KeypointRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs,draw_keypoints
# from torchvision.utils import draw_keypoints

def create_model(num_classes, box_thresh=0.5,num_keypoints=50):
    backbone = resnet50_fpn_backbone()
    model = KeypointRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh,
                     num_keypoints = num_keypoints)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 1  # 不包含背景
    box_thresh = 0.5
    num_keypoints = 50
    weights_path = "./save_weights/model_19.pth"
    img_path_list = glob.glob("/911G/data/semi_care_data/middle_down_wai/whole/train/*.jpg")
    # img_path = "./test.jpg"
    label_json_path = './coco91_indices.json'

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh,num_keypoints = num_keypoints)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    # model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)


    


    

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        # img_height, img_width = img.shape[-2:]

        img_height, img_width = (720,1280)
        init_img = torch.zeros((1, 3, img_height, img_width), device="cpu")
        model(init_img)
        

        
        count = 0
        for img_path in img_path_list:
            
            # load image
            assert os.path.exists(img_path), f"{img_path} does not exits."
            original_img = Image.open(img_path).convert('RGB')

            # from pil image to tensor, do not normalize image
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            # t_start = time_synchronized()
            t_start  = time.time()
            predictions = model(img)[0]
            # t_end = time_synchronized()
            t_end = time.time()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            # predict_keypoints = predictions["keypoints"]
            # predict_kp_scores = predictions["keypoints_scores"]
            predict_keypoints = predictions["keypoints"].squeeze().to("cpu").numpy()
            # predict_kp_scores = predictions["keypoints_scores"].squeeze().to("cpu").numpy()
            # predict_mask = predictions["masks"].to("cpu").numpy()
            # predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
                return
            import cv2
            import copy
            testimg = cv2.imread(img_path)
            # draw_img = img.squeeze().permute(1,2,0).to("cpu").numpy()
            # new_image = copy.deepcopy(draw_img)
            # key_line_range
            for i,key in enumerate(np.round(predict_keypoints)):
                
                cv2.circle(testimg,(int(key[0]),int(key[1])),5,(0,0,255),-1,8,0)
                if i>0 and i < 9:
                    cv2.line(testimg,(int(last_key[0]),int(last_key[1])),(int(key[0]),int(key[1])),(0,255,255),1,8,0)
                elif i>9 and i <18:
                    cv2.line(testimg,(int(last_key[0]),int(last_key[1])),(int(key[0]),int(key[1])),(0,255,255),1,8,0)
                elif i>18 and i < 34:
                    cv2.line(testimg,(int(last_key[0]),int(last_key[1])),(int(key[0]),int(key[1])),(255,0,0),1,8,0)
                elif i>34 and i < 50:
                    cv2.line(testimg,(int(last_key[0]),int(last_key[1])),(int(key[0]),int(key[1])),(0,0,255),1,8,0)
                last_key = key
            cv2.imshow("image",testimg)
            if not os.path.exists("./output"):
                os.makedirs("./output")
            # cv2.imwrite(os.path.join("./output",f"{count}.jpg"}),testimg)
            cv2.imwrite(os.path.join("./output",f"{count}.jpg"),testimg)
            cv2.waitKey(1)
            count+=1
            # cv2.destroyAllWindows()
        # plot_img = draw_keypoints(img[0].to(dtype= torch.uint8,device = device),predict_keypoints,colors="blue", radius=3)
        # plot_img = draw_objs(original_img,
        #                      boxes=predict_boxes,
        #                      classes=predict_classes,
        #                      scores=predict_scores,
        #                      masks=predict_mask,
        #                      category_index=category_index,
        #                      line_thickness=3,
        #                      font='arial.ttf',
        #                      font_size=20)
        # plt.imshow(plot_img)
        # plt.show()
        # # 保存预测的图片结果
        # plot_img.save("test_result.jpg")


if __name__ == '__main__':
    main()

