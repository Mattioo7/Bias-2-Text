"""
B2T (Bias-to-Text) pipeline:
    1. Load the validation dataset
    2. Classify the images with the pretrained model (cached in result/, independent of captions)
    3. Extract captions for the images the score needs (all for CLIP, misclassified for VQA)
    4. Extract keywords from captions of misclassified images, per class
    5. Score the keywords (CLIP score from the paper, or VQA score) and save them

Outputs go to outputs/. The directory path of each step holds only the arguments that step
depends on, so steps 2-4 are computed once per configuration and reused by later runs:

    outputs/<dataset>[_<celeba_variant>]/
      _cache/captions/<captioning_model>/<image>.txt   caption pool, shared by all models and subsets
      <model>/<n_all | n<N>>/
        predictions.csv                                step 2
        <captioning_model>/
          captions.csv                                 step 3, captions of this experiment's images
          <keyword_extraction_model>/
            keywords.json                              step 4
            runs/<timestamp>_<score>[_<vqa_model>]/    step 5, a new directory on every run
              config.json, diff_<class>.csv | vqa_scores.csv
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#MR to prevent an OpenMP error

# Started before the imports, so the total time includes loading the models.
import time
start_time = time.perf_counter()

# Imports below load CLIP and ClipCap, which takes a while with no output.
# DataLoader workers on Windows re-import this file as "__mp_main__", so they stay silent.
if __name__ == "__main__":
    print("B2T started. Loading libraries and models (CLIP, ClipCap)...\n", flush=True)

# CLIP ships its checkpoints as TorchScript modules, so clip.load() goes through
# the deprecated torch.jit.load. Filtered here, before any import that loads CLIP.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r"`torch\.jit\.load` is deprecated")

# ignore SourceChangeWarning when loading model
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

import argparse
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# for loading dataset
from data.celeba import CelebA, get_transform_celeba
from data.waterbirds import Waterbirds, get_transform_cub

# for various functions
from function.extract_caption import extract_caption ## default-> cuda:0/ clip:ViT-B/32
from function.extract_keyword import extract_keyword
from function.gpt_keywords import extract_gpt_keywords
from function.gpt_models import GPT_MODELS
from function.calculate_similarity import calc_similarity
from function.print_similarity import print_similarity
from function.vqa_score import calculate_vqa_score

all_captioning_models = ["clipcap", "multicap", *GPT_MODELS]
all_keyword_extraction_models = ["yake", *GPT_MODELS]
all_datasets = ['waterbird', 'celeba']
all_scores = ['clip', 'vqa']
all_vqa_models = [*GPT_MODELS, 'random']
# CelebA ships two image sets under data/celeba/; both share the annotation CSVs.
celeba_variant_dirs = {'align': 'img_align_celeba', 'raw': 'img_celeba'}

outputs_dir = 'outputs/'
model_dir = 'model/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default = 'waterbird', choices=all_datasets, help="dataset") #celeba, waterbird
    parser.add_argument("--model", type=str, default='best_model_Waterbirds_erm.pth') #best_model_CelebA_erm.pth, best_model_CelebA_dro.pth, best_model_Waterbirds_erm.pth, best_model_Waterbirds_dro.pth
    parser.add_argument("--captioning_model", type=str, default='clipcap', choices=all_captioning_models)
    parser.add_argument("--keyword_extraction_model", type=str, default='yake', choices=all_keyword_extraction_models)
    parser.add_argument("--score", type=str, default='clip', choices=all_scores)
    parser.add_argument("--vqa_model", type=str, default='gpt-4o-mini', choices=all_vqa_models,
                        help="Model answering which keywords are visible in an image (only with --score vqa). 'random' answers 0/1 at random, without the API.")
    parser.add_argument("--number_val_images", type=int, default=None, help="How many images should be used from the original val dataset. This reduces time and costs if a low number is chosen. None uses all images")
    parser.add_argument("--celeba_variant", type=str, default='align', choices=list(celeba_variant_dirs),
                        help="Which CelebA image set to use: 'align' (178x218 crops, matches the pretrained checkpoints) or 'raw' (in-the-wild originals). Ignored for waterbird.")
    parser.add_argument("--save_result", default = True)
    args = parser.parse_args()
    return args


def print_step(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_config(args, device):
    print("-" * 20 + " CONFIG " + "-" * 20)
    print(f"{'device':<26} {device}")
    for name, value in vars(args).items():
        print(f"{name:<26} {value}")
    print("-" * 48)


def output_paths(args):
    """Where each step stores its result; see the module docstring for the layout."""
    dataset_dir = outputs_dir + args.dataset
    if args.dataset == 'celeba':
        dataset_dir += "_" + args.celeba_variant
    subset = "n_all" if args.number_val_images is None else f"n{args.number_val_images}"
    predictions_dir = f"{dataset_dir}/{args.model.split('.')[0]}/{subset}"
    captioning_dir = f"{predictions_dir}/{args.captioning_model}"
    keywords_dir = f"{captioning_dir}/{args.keyword_extraction_model}"

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.score
    if args.score == "vqa":
        run_name += "_" + args.vqa_model

    paths = {
        "caption_pool": f"{dataset_dir}/_cache/captions/{args.captioning_model}/",
        "predictions": f"{predictions_dir}/predictions.csv",
        "captions": f"{captioning_dir}/captions.csv",
        "keywords": f"{keywords_dir}/keywords.json",
        "run": f"{keywords_dir}/runs/{run_name}/",
    }
    # The run directory is created in step 5, so a run that fails earlier leaves no empty one behind.
    for directory in (paths["caption_pool"], keywords_dir):
        os.makedirs(directory, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# Step 1: dataset
# ---------------------------------------------------------------------------
def load_dataset(args):
    """Returns (val_dataset, class_names, image_dir)."""
    if args.dataset == 'waterbird':
        preprocess = get_transform_cub()
        class_names = ['landbird', 'waterbird']
        # group_names = ['landbird_land', 'landbird_water', 'waterbird_land', 'waterbird_water']
        image_dir = 'data/cub/data/waterbird_complete95_forest2water2/'
        val_dataset = Waterbirds(data_dir='data/cub/data/waterbird_complete95_forest2water2', split='val', transform=preprocess)
    elif args.dataset == 'celeba':
        preprocess = get_transform_celeba()
        class_names = ['not blond', 'blond']
        # group_names = ['not blond_female', 'not blond_male', 'blond_female', 'blond_male']
        variant_dir = celeba_variant_dirs[args.celeba_variant]
        if args.celeba_variant == 'raw':
            print("WARNING: get_transform_celeba() center-crops to 178px, which only makes sense for the "
                  "aligned images. The pretrained checkpoints were trained on those, so results on 'raw' "
                  "are not comparable to the paper.")
        image_dir = f'data/celeba/{variant_dir}/data/'
        val_dataset = CelebA(data_dir='data/celeba', split='val', transform=preprocess, variant=variant_dir)
    else:
        raise ValueError(f"Dataset must be within {all_datasets}, but was {args.dataset}")

    if args.number_val_images is not None:
        # ensure that the given number is not too large
        num_imgs = min(args.number_val_images, len(val_dataset))
        print(f"-------- LIMIT DATASET TO {num_imgs} IMAGES TO REDUCE COSTS!!! Originally {len(val_dataset)} images -----------")
        val_dataset = Subset(val_dataset, range(num_imgs))

    print(f"Validation images: {len(val_dataset)}")
    return val_dataset, class_names, image_dir


# ---------------------------------------------------------------------------
# Step 2: predictions
# ---------------------------------------------------------------------------
def classify(val_dataset, model_name, device):
    """Runs the classifier. Returns one row per image, without captions."""
    val_dataloader = DataLoader(val_dataset, batch_size=256, num_workers=4, drop_last=False)

    model = torch.load(model_dir + model_name, weights_only=False)
    model = model.to(device)
    model.eval()
    print("Pretrained model \"{}\" loaded".format(model_name))

    result = {"image":[],
            "pred":[],
            "actual":[],
            "group":[],
            "spurious":[],
            "correct":[],
            }

    with torch.no_grad():
        for (images, (targets, targets_g, targets_s), index, paths) in tqdm(val_dataloader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for i in range(len(preds)):
                result['image'].append(paths[i])
                result['pred'].append(preds[i].item())
                result['actual'].append(targets[i].item())
                result['group'].append(targets_g[i].item())
                result['spurious'].append(targets_s[i].item())
                result['correct'].append(int(preds[i] == targets[i]))

    return pd.DataFrame(result)


def load_or_classify(val_dataset, predictions_path, model_name, device):
    """Predictions depend only on the model and the images, so they are computed once and cached."""
    if os.path.exists(predictions_path):
        df = pd.read_csv(predictions_path)
        # Caches written before the reordering also held captions and the pandas index.
        df = df.drop(columns=[c for c in ("caption", "Unnamed: 0") if c in df.columns])
        if len(df) == len(val_dataset):
            print("Predictions \"{}\" loaded".format(predictions_path))
            return df
        print(f"Predictions \"{predictions_path}\" have {len(df)} rows but the dataset has "
              f"{len(val_dataset)} images. Classifying again.")

    df = classify(val_dataset, model_name, device)
    df.to_csv(predictions_path, index=False)
    print("Predictions stored in \"{}\"".format(predictions_path))
    return df


def print_accuracy(df):
    running_corrects = int(df['correct'].sum())
    print("# of correct examples : ", running_corrects)
    print("# of wrong examples : ", len(df) - running_corrects)
    print("# of all examples : ", len(df))
    print("Accuracy : {:.2f} %".format(running_corrects/len(df)*100))


# ---------------------------------------------------------------------------
# Step 3: captions
# ---------------------------------------------------------------------------
def caption_path_for(caption_dir, image):
    return caption_dir + image.split("/")[-1][:-4] + ".txt"


def extract_captions(images, image_dir, caption_dir, captioning_model):
    """Writes one caption .txt per image into caption_dir, skipping images an earlier run already captioned."""
    missing = [image for image in images if not os.path.exists(caption_path_for(caption_dir, image))]
    print(f"{len(images) - len(missing)} captions loaded from '{caption_dir}', {len(missing)} to extract")
    if not missing:
        return
    for image in tqdm(missing):
        caption = extract_caption(image_dir + image, captioning_model)
        with open(caption_path_for(caption_dir, image), 'w') as f:
            f.write(caption)
    print("Captions of {} images extracted".format(len(missing)))


def read_captions(images, caption_dir):
    captions = []
    for image in images:
        path = caption_path_for(caption_dir, image)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Caption '{path}' is missing.")
        with open(path, "r") as f:
            captions.append(f.readline())
    return captions


def save_captions(df, caption_dir, captions_path):
    """Captions of every image of this experiment found in the pool. Rewritten on each run, so a
    CLIP run (all images) after a VQA run (misclassified only) extends the file."""
    images = [image for image in df['image'] if os.path.exists(caption_path_for(caption_dir, image))]
    pd.DataFrame({"image": images, "caption": read_captions(images, caption_dir)}).to_csv(captions_path, index=False)
    print(f"Captions of {len(images)} images stored in \"{captions_path}\"")


# ---------------------------------------------------------------------------
# Step 4: keywords
# ---------------------------------------------------------------------------
def extract_keywords(df_wrong_class, keyword_extraction_model):
    """Keywords from the joined captions of one class's misclassified images."""
    captions = ' '.join(df_wrong_class['caption'].tolist())
    if keyword_extraction_model in GPT_MODELS:
        return extract_gpt_keywords(captions, model=keyword_extraction_model)
    if keyword_extraction_model == "yake":
        return extract_keyword(captions)
    raise ValueError(f"Unknown keyword extraction model: '{keyword_extraction_model}'")


def load_or_extract_keywords(keywords_path, df_wrong, class_names, keyword_extraction_model):
    """Keywords are extracted once per experiment and reused, which also keeps GPT keywords fixed across runs."""
    if os.path.exists(keywords_path):
        with open(keywords_path) as f:
            saved = json.load(f)
        print("Keywords \"{}\" loaded".format(keywords_path))
        return {c: saved[class_names[c]] for c in (0, 1)}

    keywords = {c: extract_keywords(df_wrong[c], keyword_extraction_model) for c in (0, 1)}
    empty = [class_names[c] for c in (0, 1) if not keywords[c]]
    if empty:
        # extract_gpt_keywords returns [] when it can't parse the answer; caching that would break every later run.
        print(f"WARNING: no keywords for {empty}. Not storing them, the next run extracts them again.")
    else:
        with open(keywords_path, 'w') as f:
            json.dump({class_names[c]: keywords[c] for c in (0, 1)}, f, indent=2)
        print("Keywords stored in \"{}\"".format(keywords_path))
    return keywords


# ---------------------------------------------------------------------------
# Step 5: scores
# ---------------------------------------------------------------------------
def save_config(args, device, paths):
    config = {**vars(args), "device": str(device), "paths": paths}
    config_path = paths["run"] + "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Run directory: \"{paths['run']}\"")


def vqa_score(image_dir, df_correct, df_wrong, keywords, vqa_model, save_result, run_dir):
    stats = {}
    for c in (0, 1):
        print(f"\nCalculate VQA score for class {c}")
        img_paths_correct = [image_dir + image for image in df_correct[c]['image'].to_list()]
        img_paths_wrong = [image_dir + image for image in df_wrong[c]['image'].to_list()]
        stats[c] = calculate_vqa_score(img_paths_correct, img_paths_wrong, keywords[c], model=vqa_model)

    if save_result:
        df_result = pd.DataFrame({
            'Metric': [
                'correct_ratios', 'wrong_ratios',
                'correct_keyword_occurrences', 'wrong_keyword_occurrences',
                'total_correct', 'total_wrong'
            ],
            'Class_0': list(stats[0]),
            'Class_1': list(stats[1]),
        })
        output_path = run_dir + 'vqa_scores.csv'
        df_result.to_csv(output_path, index=False)
        print(f"\nData saved to {output_path}")


def clip_score(args, image_dir, class_names, df_class, df_correct, df_wrong, keywords, run_dir):
    """CLIP score from the paper: similarity on wrong images minus similarity on correct ones."""
    if args.number_val_images is not None:
        print("Warning: If all images are classified correctly, then score calculation will throw an error")
    dist = {}
    for c in (0, 1):
        print(f"\nCLIP similarity for class '{class_names[c]}'")
        print(f"  wrong images ({len(df_wrong[c])}):")
        similarity_wrong = calc_similarity(image_dir, df_wrong[c]['image'], keywords[c])
        print(f"  correct images ({len(df_correct[c])}):")
        similarity_correct = calc_similarity(image_dir, df_correct[c]['image'], keywords[c])
        dist[c] = similarity_wrong - similarity_correct

    diffs = {}
    for c, other in ((0, 1), (1, 0)):
        print("\n" + "*"*60)
        print("Result for class :", class_names[c])
        print("*"*60 + "\n")
        diffs[c] = print_similarity(keywords[c], keywords[other], dist[c], dist[other], df_class[c])

    if args.save_result:
        print()
        for c in (0, 1):
            diff_path = run_dir + "diff_" + class_names[c] + ".csv"
            diffs[c].to_csv(diff_path)
            print(f"Data saved to {diff_path}")


if __name__ == "__main__":  #MR added this to prevent an error

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    print_config(args, device)

    paths = output_paths(args)
    caption_dir = paths["caption_pool"]

    print_step("STEP 1/5: Load dataset")
    val_dataset, class_names, image_dir = load_dataset(args)

    print_step("STEP 2/5: Classify images")
    df = load_or_classify(val_dataset, paths["predictions"], args.model, device)
    print_accuracy(df)

    print_step("STEP 3/5: Extract captions")
    # Keywords come from misclassified images only. The CLIP score also looks the keywords up
    # in the captions of the correctly classified images (Acc. column), VQA does not.
    needs_caption = df['correct'] == 0 if args.score == "vqa" else pd.Series(True, index=df.index)
    images_to_caption = df.loc[needs_caption, 'image'].tolist()
    print(f"Captions needed for {len(images_to_caption)} of {len(df)} images ({args.score.upper()} score)")
    extract_captions(images_to_caption, image_dir, caption_dir, args.captioning_model)
    df.loc[needs_caption, 'caption'] = read_captions(images_to_caption, caption_dir)
    save_captions(df, caption_dir, paths["captions"])

    # Split into class 0 (landbird / not blond) and class 1 (waterbird / blond)
    df_class = {c: df[df['actual'] == c] for c in (0, 1)}
    df_correct = {c: df_class[c][df_class[c]['correct'] == 1] for c in (0, 1)}
    df_wrong = {c: df_class[c][df_class[c]['correct'] == 0] for c in (0, 1)}

    print_step("STEP 4/5: Extract keywords")
    keywords = load_or_extract_keywords(paths["keywords"], df_wrong, class_names, args.keyword_extraction_model)

    print_step(f"STEP 5/5: Calculate {args.score.upper()} score")
    if args.save_result:
        os.makedirs(paths["run"])
        save_config(args, device, paths)
    if args.score == "vqa":
        vqa_score(image_dir, df_correct, df_wrong, keywords, args.vqa_model, args.save_result, paths["run"])
    else:  # if not otherwise specified use CLIP score from paper
        clip_score(args, image_dir, class_names, df_class, df_correct, df_wrong, keywords, paths["run"])

    elapsed = int(time.perf_counter() - start_time)
    print(f"\nB2T finished in {elapsed // 3600}h {elapsed % 3600 // 60}m {elapsed % 60}s")
