from tqdm import tqdm
import os
import torch
import time
import pandas as pd

from data.waterbirds import Waterbirds, get_transform_cub
from function.gpt_keywords import extract_gpt_keywords
from function.extract_keyword import extract_keyword
from function.vqa_score import calculate_vqa_score # typo vsq


if __name__ == "__main__":
    # For debugging purposes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ### THESE VARIABLES MUST BE SET
    model = 'best_model_Waterbirds_erm.pth'
    keyword_extraction_model = "gpt-4o-mini"
    caption_dir = 'data/cub/caption_gpt-4o-mini/'
    ###

    ### THE FOLLOWING CODE READS CAPTIONS, EXTRACTS KEYWORDS AND CLASSIFIES IMAGES
    dataset = 'waterbird'  # this script works only for Waterbirds right now
    class_names = ['landbird', 'waterbird']
    preprocess = get_transform_cub()
    image_dir = 'data/cub/data/waterbird_complete95_forest2water2/'
    if not os.path.exists(caption_dir):
        os.makedirs(caption_dir)
        print(f"Directory '{caption_dir}' created.")
    else:
        print(f"Directory '{caption_dir}' already exists. Writing content into or reading content from this directory")
    val_dataset = Waterbirds(data_dir='data/cub/data/waterbird_complete95_forest2water2', split='val',
                             transform=preprocess)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers=4, drop_last=False)

    result_dir = 'result_test/'  # 'result_gpt-4o-mini_2/'
    model_dir = 'model/'
    diff_dir = 'diff_test/'  # 'diff_gpt-4o-mini_2/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(diff_dir):
        os.makedirs(diff_dir)

    # correctify dataset
    result_path = result_dir + dataset + "_" + model.split(".")[0] + ".csv"
    if not os.path.exists(result_path):
        model = torch.load(model_dir + model, weights_only=False)
        model = model.to(device)
        model.eval()
        start_time = time.time()
        print("Pretrained model \"{}\" loaded".format(model))

        result = {"image": [],
                  "pred": [],
                  "actual": [],
                  "group": [],
                  "spurious": [],
                  "correct": [],
                  "caption": [],
                  }

        with torch.no_grad():
            running_corrects = 0
            for (images, (targets, targets_g, targets_s), index, paths) in tqdm(val_dataloader):
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                for i in range(len(preds)):
                    image = paths[i]
                    pred = preds[i]
                    actual = targets[i]
                    group = targets_g[i]
                    spurious = targets_s[i]
                    caption_path = caption_dir + image.split("/")[-1][:-4] + ".txt"
                    with open(caption_path, "r") as f:
                        caption = f.readline()
                    result['image'].append(image)
                    result['pred'].append(pred.item())
                    result['actual'].append(actual.item())
                    result['group'].append(group.item())
                    result['spurious'].append(spurious.item())
                    result['caption'].append(caption)
                    if pred == actual:
                        result['correct'].append(1)
                        running_corrects += 1
                    else:
                        result['correct'].append(0)

            print("# of correct examples : ", running_corrects)
            print("# of wrong examples : ", len(val_dataset) - running_corrects)
            print("# of all examples : ", len(val_dataset))
            print("Accuracy : {:.2f} %".format(running_corrects / len(val_dataset) * 100))

        df = pd.DataFrame(result)
        df.to_csv(result_path)
        print("Classified result stored")
    else:
        df = pd.read_csv(result_path)
        print("Classified result \"{}\" loaded".format(result_path))

    df_wrong = df[df['correct'] == 0]
    df_correct = df[df['correct'] == 1]
    df_class_0 = df[df['actual'] == 0]  # not blond, landbird
    df_class_1 = df[df['actual'] == 1]  # blond, waterbird
    df_wrong_class_0 = df_wrong[df_wrong['actual'] == 0]
    df_wrong_class_1 = df_wrong[df_wrong['actual'] == 1]
    df_correct_class_0 = df_correct[df_correct['actual'] == 0]
    df_correct_class_1 = df_correct[df_correct['actual'] == 1]

    caption_wrong_class_0 = ' '.join(df_wrong_class_0['caption'].tolist())
    caption_wrong_class_1 = ' '.join(df_wrong_class_1['caption'].tolist())

    # extract keyword
    if "gpt" in keyword_extraction_model:
        keywords_class_0 = extract_gpt_keywords(caption_wrong_class_0)
        keywords_class_1 = extract_gpt_keywords(caption_wrong_class_1)
    else:
        # use yake if not otherwise specified
        keywords_class_0 = extract_keyword(caption_wrong_class_0)
        keywords_class_1 = extract_keyword(caption_wrong_class_1)
    all_keywords = [keywords_class_0, keywords_class_1]
    ###

    ### HERE WE START WITH THE ACTUAL VQA SCORE CALCULATION

    # manually limit to n images to reduce costs -> n images per class and correct/wrong prediction
    n = 2
    n_images_correct_0 = df_correct_class_0['image'].to_list()[:n]
    img_paths_correct_0 = [image_dir + image for image in n_images_correct_0]
    n_images_correct_1 = df_correct_class_1['image'].to_list()[:n]
    img_paths_correct_1 = [image_dir + image for image in n_images_correct_1]
    n_images_wrong_0 = df_wrong_class_0['image'].to_list()[:n]
    img_paths_wrong_0 = [image_dir + image for image in n_images_wrong_0]
    n_images_wrong_1 = df_wrong_class_1['image'].to_list()[:n]
    img_paths_wrong_1 = [image_dir + image for image in n_images_wrong_1]

    img_paths_correct = [img_paths_correct_0, img_paths_correct_1]
    img_paths_wrong = [img_paths_wrong_0, img_paths_wrong_1]
    print("img_paths_correct:\n", img_paths_correct)

    for name, corr_images, wrong_images, keywords in zip(class_names, img_paths_correct, img_paths_wrong, all_keywords):
        correct_ratios, wrong_ratios = calculate_vqa_score(corr_images, wrong_images, keywords)

        print(f"Correct Ratios of {name}: {correct_ratios}")
        print(f"Wrong Ratios of {name}: {wrong_ratios}")
    ###
