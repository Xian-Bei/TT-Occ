#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automatic Grounding Data Engine

This application automatically generates grounding annotations from image captions:
1. Extract noun phrases from captions using spaCy
2. Ground each phrase using Rex-Omni
3. Generate annotations with character-level spans and bounding boxes
4. Save results to JSONL format for training data
"""

import json
import os
from typing import Dict, List, Tuple

import spacy
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from rex_omni import RexOmniVisualize, RexOmniWrapper


def setup_spacy(model: str = "en_core_web_sm"):
    """
    Initialize spaCy model for phrase extraction

    Args:
        model: spaCy model name

    Returns:
        spaCy nlp model
    """
    try:
        nlp = spacy.load(model)
    except OSError:
        print(f"⚠️  spaCy model '{model}' not found.")
        print(f"📥 Downloading model...")
        import subprocess

        subprocess.run(["python", "-m", "spacy", "download", model])
        nlp = spacy.load(model)

    return nlp


def extract_noun_phrases(caption: str, nlp) -> List[Tuple[str, int, int]]:
    """
    Extract noun phrases from caption with character spans

    Args:
        caption: Input caption text
        nlp: spaCy nlp model

    Returns:
        List of (phrase, start_char, end_char) tuples
    """
    doc = nlp(caption)
    phrases = []

    for noun_chunk in doc.noun_chunks:
        phrase = str(noun_chunk).lower().strip()
        # Get character-level span
        start_char = noun_chunk.start_char
        end_char = noun_chunk.end_char
        phrases.append((phrase, start_char, end_char))

    return phrases


def ground_phrases_with_rex(
    rex_model: RexOmniWrapper,
    image: Image.Image,
    phrases: List[str],
    task: str = "detection",
) -> Dict:
    """
    Ground phrases using Rex-Omni

    Args:
        rex_model: Rex-Omni wrapper
        image: Input image
        phrases: List of phrases to ground
        task: Task type

    Returns:
        Grounding results
    """
    if not phrases:
        return {}

    # Remove duplicates while preserving order
    unique_phrases = []
    seen = set()
    for phrase in phrases:
        if phrase not in seen:
            unique_phrases.append(phrase)
            seen.add(phrase)

    print(f"🔍 Grounding {len(unique_phrases)} unique phrases...")

    results = rex_model.inference(images=image, task=task, categories=unique_phrases)

    result = results[0]
    if not result["success"]:
        print(f"⚠️  Grounding failed: {result.get('error', 'Unknown error')}")
        return {}

    return result["extracted_predictions"]


def create_grounding_annotation(
    image_path: str,
    caption: str,
    phrases_with_spans: List[Tuple[str, int, int]],
    grounding_results: Dict,
    image_width: int,
    image_height: int,
) -> Dict:
    """
    Create grounding annotation in JSONL format

    Args:
        image_path: Path to image
        caption: Original caption
        phrases_with_spans: List of (phrase, start_char, end_char)
        grounding_results: Rex-Omni grounding results
        image_width: Image width
        image_height: Image height

    Returns:
        Annotation dictionary
    """
    annotation = {
        "image": image_path,
        "caption": caption,
        "width": image_width,
        "height": image_height,
        "phrases": [],
    }

    # Map each phrase to its grounding boxes
    for phrase, start_char, end_char in phrases_with_spans:
        phrase_data = {
            "phrase": phrase,
            "start_char": start_char,
            "end_char": end_char,
            "boxes": [],
        }

        # Get grounding boxes for this phrase
        if phrase in grounding_results:
            for detection in grounding_results[phrase]:
                if detection["type"] == "box":
                    coords = detection["coords"]
                    # Normalize coordinates to [0, 1]
                    normalized_box = [
                        coords[0] / image_width,
                        coords[1] / image_height,
                        coords[2] / image_width,
                        coords[3] / image_height,
                    ]
                    phrase_data["boxes"].append(
                        {
                            "bbox": coords,  # Absolute coordinates
                        }
                    )

        # Only add phrases that have grounding boxes
        if phrase_data["boxes"]:
            annotation["phrases"].append(phrase_data)

    return annotation


def visualize_grounding(image: Image.Image, annotation: Dict, save_path: str):
    """
    Visualize grounding annotations

    Args:
        image: Input image
        annotation: Annotation dictionary
        save_path: Path to save visualization
    """
    # Create a copy for visualization
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)

    # Try to use a truetype font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Use different colors for different phrases
    import matplotlib.pyplot as plt

    colors = plt.cm.tab20(range(20))

    # Draw boxes and labels for each phrase
    for idx, phrase_data in enumerate(annotation["phrases"]):
        phrase = phrase_data["phrase"]
        color_idx = idx % 20
        color = tuple((colors[color_idx][:3] * 255).astype(int).tolist())

        for box_data in phrase_data["boxes"]:
            bbox = box_data["bbox"]
            x1, y1, x2, y2 = bbox

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label background
            label_bbox = draw.textbbox((x1, y1 - 20), phrase, font=font)
            draw.rectangle(label_bbox, fill=color)

            # Draw label text
            draw.text((x1, y1 - 20), phrase, fill=(255, 255, 255), font=font)

    # Add caption at the bottom
    caption = annotation["caption"]
    # Wrap caption if too long
    max_width = image.width - 20
    words = caption.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    # Draw caption box at bottom
    caption_height = len(lines) * 25 + 20
    draw.rectangle(
        [0, image.height - caption_height, image.width, image.height],
        fill=(0, 0, 0, 180),
    )

    # Draw caption text
    y_offset = image.height - caption_height + 10
    for line in lines:
        draw.text((10, y_offset), line, fill=(255, 255, 255), font=font)
        y_offset += 25

    # Save visualization
    vis_image.save(save_path)
    print(f"💾 Visualization saved to: {save_path}")


def process_single_image(
    image_path: str,
    caption: str,
    rex_model: RexOmniWrapper,
    nlp,
    output_dir: str,
    visualize: bool = True,
) -> Dict:
    """
    Process a single image to generate grounding annotations

    Args:
        image_path: Path to image
        caption: Image caption
        rex_model: Rex-Omni model
        nlp: spaCy model
        output_dir: Output directory
        visualize: Whether to save visualization

    Returns:
        Annotation dictionary
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    # Extract noun phrases with character spans
    phrases_with_spans = extract_noun_phrases(caption, nlp)

    if not phrases_with_spans:
        print(f"⚠️  No phrases extracted from caption")
        return None

    print(f"📝 Extracted {len(phrases_with_spans)} phrases")

    # Ground phrases
    phrases = [p[0] for p in phrases_with_spans]
    grounding_results = ground_phrases_with_rex(rex_model, image, phrases)

    # Create annotation
    annotation = create_grounding_annotation(
        image_path=image_path,
        caption=caption,
        phrases_with_spans=phrases_with_spans,
        grounding_results=grounding_results,
        image_width=image_width,
        image_height=image_height,
    )

    # Visualize if requested
    if visualize and annotation["phrases"]:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(output_dir, f"{image_name}_grounding.jpg")
        visualize_grounding(image, annotation, vis_path)

    return annotation


def batch_process_images(
    image_caption_pairs: List[Tuple[str, str]],
    rex_model_path: str = "IDEA-Research/Rex-Omni",
    backend: str = "transformers",
    output_jsonl: str = "grounding_annotations.jsonl",
    output_dir: str = "grounding_visualizations",
    visualize: bool = True,
):
    """
    Batch process multiple images to generate grounding data

    Args:
        image_caption_pairs: List of (image_path, caption) tuples
        rex_model_path: Rex-Omni model path
        backend: Rex-Omni backend
        output_jsonl: Output JSONL file path
        output_dir: Output directory for visualizations
        visualize: Whether to save visualizations
    """
    print("=" * 60)
    print("Automatic Grounding Data Engine")
    print("=" * 60)

    # Setup models
    print("\n🤖 Initializing models...")
    nlp = setup_spacy()
    print("✅ spaCy model loaded!")

    rex_model = RexOmniWrapper(
        model_path=rex_model_path,
        backend=backend,
        max_tokens=2048,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )
    print("✅ Rex-Omni model loaded!")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process images
    annotations = []
    print(f"\n📊 Processing {len(image_caption_pairs)} images...")

    for image_path, caption in tqdm(image_caption_pairs):
        print(f"\n📷 Processing: {image_path}")
        print(f"💬 Caption: {caption[:100]}...")

        try:
            annotation = process_single_image(
                image_path=image_path,
                caption=caption,
                rex_model=rex_model,
                nlp=nlp,
                output_dir=output_dir,
                visualize=visualize,
            )

            if annotation and annotation["phrases"]:
                annotations.append(annotation)
                print(f"✅ Grounded {len(annotation['phrases'])} phrases")
            else:
                print(f"⚠️  No phrases grounded for this image")

        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            continue

    # Save annotations to JSONL
    print(f"\n💾 Saving annotations to {output_jsonl}...")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for annotation in annotations:
            f.write(json.dumps(annotation, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(annotations)} annotations!")
    print(f"📁 Annotations: {output_jsonl}")
    print(f"📁 Visualizations: {output_dir}/")
    print("\n" + "=" * 60)
    print("✅ Data generation completed!")
    print("=" * 60)


def main():
    """
    Example usage of the grounding data engine
    """
    # Example data
    image_caption_pairs = [
        (
            "tutorials/detection_example/test_images/cafe.jpg",
            "The image shows a modern, vibrant office space with people interacting in a collaborative environment. There are a mix of individuals sitting, working, and socializing around a stylish lounge area with large yellow flowers on a coffee table. The room features a sleek kitchen area with bright yellow pendant lights, and there's a creative neon artwork of a robot on the wall.",
        ),
        (
            "tutorials/detection_example/test_images/boys.jpg",
            "Three young boys are in a music room with guitars on the wall. One boy is holding a microphone while another is playing a piano.",
        ),
    ]

    # Run batch processing
    batch_process_images(
        image_caption_pairs=image_caption_pairs,
        rex_model_path="IDEA-Research/Rex-Omni",
        backend="transformers",
        output_jsonl="applications/_2_automatic_grounding_data_engine/grounding_annotations.jsonl",
        output_dir="applications/_2_automatic_grounding_data_engine/visualizations",
        visualize=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automatic Grounding Data Engine")
    parser.add_argument("--images", nargs="+", help="List of image paths")
    parser.add_argument(
        "--captions", nargs="+", help="List of captions (must match images)"
    )
    parser.add_argument(
        "--input-json", type=str, help="Input JSON file with image-caption pairs"
    )
    parser.add_argument(
        "--rex-model",
        type=str,
        default="IDEA-Research/Rex-Omni",
        help="Rex-Omni model path",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Rex-Omni backend",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="grounding_annotations.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="grounding_visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--no-visualize", action="store_true", help="Don't save visualizations"
    )

    args = parser.parse_args()

    # Prepare image-caption pairs
    image_caption_pairs = []

    if args.input_json:
        # Load from JSON file
        with open(args.input_json, "r") as f:
            data = json.load(f)
        image_caption_pairs = [(item["image"], item["caption"]) for item in data]

    elif args.images and args.captions:
        # Use command line arguments
        if len(args.images) != len(args.captions):
            raise ValueError("Number of images and captions must match")
        image_caption_pairs = list(zip(args.images, args.captions))

    else:
        # Use default examples
        print("No input provided, running with default examples...")
        main()
        exit(0)

    # Run batch processing
    batch_process_images(
        image_caption_pairs=image_caption_pairs,
        rex_model_path=args.rex_model,
        backend=args.backend,
        output_jsonl=args.output_jsonl,
        output_dir=args.output_dir,
        visualize=not args.no_visualize,
    )
