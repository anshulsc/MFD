import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
import time 
import os 
import argparse
from termcolor import cprint

from src.models import RED_DOT
from src.utils import modality_fusion, prepare_input 

from lavis.models import load_model_and_preprocess

LABEL_MAP = {0: "True", 1: "Fake"}


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--emb-dim', type=int, default=768)
    parser.add_argument('--tf-layers', type=int, default=4)
    parser.add_argument('--tf-head', type=int, default=8)
    parser.add_argument('--tf-dim', type=int, default=128)
    parser.add_argument('--use-evidence', type=int, default=0)
    parser.add_argument('--use-neg-evidence', type=int, default=0)
    parser.add_argument('--model-version', type=str, default='baseline')

    parser.add_argument('--checkpoint-path', type=str, default='checkpoints_pt/best_model.pt')
    parser.add_argument('--test-data-path', type=str, default='Test Data')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    return parser.parse_args()



def load_models(args, device):
    
    cprint("Loading CLIP model for feature extraction...", "cyan")
    clip_model, vis_processors, txt_processors = load_model_and_preprocess(
        name="clip_feature_extractor", 
        model_type="ViT-L-14", 
        is_eval=True, 
        device=device
    )
    cprint("CLIP model loaded.", "green")

    cprint("Loading RED-DOT model for prediction...", "cyan")
    model = RED_DOT(
        device=device,
        emb_dim=args.emb_dim,
        tf_layers=args.tf_layers,
        tf_head=args.tf_head,
        tf_dim=args.tf_dim,
        use_evidence=args.use_evidence,
        use_neg_evidence=args.use_neg_evidence,
        model_version=args.model_version
    )

    cprint(f"Loading checkpoint from: {args.checkpoint_path}", "cyan")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    model.to(device)
    cprint("RED-DOT model loaded and ready for inference.", "green")
    
    return clip_model, vis_processors, txt_processors, model



def predict(image_path, caption, clip_model, vis_processors, txt_processors, red_dot_model, device):
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None


    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text = txt_processors["eval"](caption)
    sample = {"image": image, "text_input": [text]}
    

    clip_features = clip_model.extract_features(sample, mode="multimodal")
    image_embedding = clip_features.image_embeds_proj[:, 0]
    text_embedding = clip_features.text_embeds_proj[:, 0]

    fusion_method = ["concat_1", "add", "sub", "mul"]
    model_input = prepare_input(
        fusion_method=fusion_method,
        fuse_evidence=[False],
        use_evidence=0,
        images=image_embedding,
        texts=text_embedding
    )

    with torch.no_grad():
        output_logit, _ = red_dot_model(model_input, inference=True)


    confidence_score = torch.sigmoid(output_logit).item()
    predicted_idx = (confidence_score > 0.5).astype(int)
    predicted_label = LABEL_MAP[predicted_idx]


    p = confidence_score
    entropy_score = 0.0 if p in [0, 1] else - (p * np.log2(p) + (1 - p) * np.log2(1 - p))

    return {
        "predicted_label": predicted_label,
        "confidence_score": confidence_score,
        "entropy_score": entropy_score
    }

def analyze_test_data(args, clip_model, vis_processors, txt_processors, red_dot_model, device):
    results = []
    
    print("\n" + "="*50)
    print(f"Running Analysis on Test Data from: {args.test_data_path}")
    print("="*50)

    if not os.path.isdir(args.test_data_path):
        print(f"Error: Test data directory not found at '{args.test_data_path}'")
        return

    for sample_dir in sorted(os.listdir(args.test_data_path)):
        sample_path = os.path.join(args.test_data_path, sample_dir)
        if not os.path.isdir(sample_path):
            continue

        image_file = os.path.join(sample_path, 'Image.jpg')
        caption_file = os.path.join(sample_path, 'caption.txt')
        gt_file = os.path.join(sample_path, 'GT.txt')

        if not all(os.path.exists(f) for f in [image_file, caption_file, gt_file]):
            print(f"Skipping {sample_dir}: missing files.")
            continue

        with open(caption_file, 'r', encoding='utf-8') as f:
            caption = f.read().splitlines()[0].strip()
        with open(gt_file, 'r') as f:
            ground_truth = f.read().strip().lower()

        start_time = time.time()
        result = predict(image_file, caption, clip_model, vis_processors, txt_processors, red_dot_model, device)
        inference_time = time.time() - start_time
        
        if result:
            results.append({
                "Sample": sample_dir,
                "Ground Truth": ground_truth,
                "Predicted Label": result['predicted_label'].lower(),
                "Confidence (for Fake)": f"{result['confidence_score']:.4f}",
                "Entropy": f"{result['entropy_score']:.4f}",
                "Inference Time (s)": f"{inference_time:.2f}"
            })

    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df['Correct'] = df['Ground Truth'] == df['Predicted Label']
        print(df.to_string())
        
        accuracy = df['Correct'].sum() / len(df) if len(df) > 0 else 0
        print(f"\nOverall Accuracy: {accuracy:.2%} ({df['Correct'].sum()}/{len(df)})")

    except ImportError:
        print("\nPandas not found. Printing raw results:")
        for res in results:
            print(res)
            
    print("\n" + "="*50)
    print("Analysis Complete")
    print("="*50)


if __name__ == '__main__':

    args = parse_args()


    if args.device == 'cuda' and not torch.cuda.is_available():
        cprint("CUDA is not available. Defaulting to CPU.", "yellow")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
        
    cprint(f"Using device: {device}", "green")


    clip_model, vis_processors, txt_processors, red_dot_model = load_models(args, device)
    

    analyze_test_data(args, clip_model, vis_processors, txt_processors, red_dot_model, device)
    
    cprint("\n--- Example of a single prediction ---", "cyan")
    single_image = os.path.join(args.test_data_path, 'sample1/Image.jpg')
    single_caption = "Sri Lanka war: UN council backs rights abuses inquiry"
    
    if os.path.exists(single_image):
        single_result = predict(single_image, single_caption, clip_model, vis_processors, txt_processors, red_dot_model, device)
        print(f"Input Image: {single_image}")
        print(f"Input Caption: '{single_caption}'")
        print(f"Prediction Result: {single_result}")
    else:
        print(f"Could not run single prediction example, file not found: {single_image}")
