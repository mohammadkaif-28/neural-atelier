import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision.transforms.functional import to_pil_image

from utils.utils import blend_features, calc_mean_std, inference_tensor, style_statistics_loss


def load_pairs(folder):
    folder = Path(folder)
    pairs = json.loads((folder / 'pairs.json').read_text(encoding='utf-8'))
    if not pairs:
        raise ValueError("At least one validation pair is required")
    return [(p['label'], folder / p['content'], folder / p['style']) for p in pairs]


def preview_image(image):
    return ImageOps.contain(image, (352, 284), method=Image.Resampling.LANCZOS)


@torch.inference_mode()
def evaluate(decoder, encoder, target, style_stats, weights):
    output = decoder(target)
    features = encoder(output)
    content = torch.nn.functional.mse_loss(features[-1], target).item()
    style = style_statistics_loss(features, style_stats).item()
    scores = dict(content=content, style=style, weighted_total=weights[0]*content + weights[1]*style)
    picture = to_pil_image(output[0].clamp(0, 1).cpu())
    return preview_image(picture), scores


class Validation:
    """Use the app's full-frame 512px preprocessing and alpha blend for both decoders."""

    @torch.inference_mode()
    def __init__(self, pairs, encoder, baseline, device, alpha, weights):
        self.encoder = encoder
        self.alpha = alpha
        self.weights = weights
        self.rows = []
        for label, photo_path, painting_path in pairs:
            with Image.open(photo_path) as source, Image.open(painting_path) as artwork:
                photo, painting = source.convert('RGB'), artwork.convert('RGB')
            content = encoder(inference_tensor(photo, device), is_test=True)
            style = encoder(inference_tensor(painting, device))
            target = blend_features(content, style[-1], alpha)
            stats = [calc_mean_std(layer) for layer in style]
            original, scores = evaluate(baseline, encoder, target, stats, weights)
            self.rows.append((label, preview_image(photo), preview_image(painting),
                              target, stats, original, scores))

    @torch.inference_mode()
    def write(self, decoder, folder, step):
        sheet = Image.new('RGB', (4 * 368, 46 + 322 * len(self.rows)), 'white')
        draw = ImageDraw.Draw(sheet)
        for column, title in enumerate(('Content', 'Painting', 'App baseline', f'Candidate | step {step}')):
            draw.text((column * 368 + 10, 10), title, fill='black', font_size=20)
        metrics = {'step': step, 'alpha': self.alpha, 'pairs': {}}
        for row, (label, photo, painting, target, stats, original, scores) in enumerate(self.rows):
            current, current_scores = evaluate(decoder, self.encoder, target, stats, self.weights)
            metrics['pairs'][label] = {'baseline': scores, 'candidate': current_scores}
            top = 46 + 322 * row
            draw.text((10, top), label, fill='black', font_size=16)
            for col, im in enumerate((photo, painting, original, current)):
                sheet.paste(im, (col * 368 + (368 - im.width)//2, top + 25 + (284 - im.height)//2))
        for model in ('baseline', 'candidate'):
            metrics[model] = {
                key: sum(p[model][key] for p in metrics['pairs'].values()) / len(self.rows)
                for key in ('content', 'style', 'weighted_total')
            }
        name = f'comparison_{step:06d}'
        image_tmp = folder / (name + '.png.tmp')
        sheet.save(image_tmp, format='PNG')
        image_tmp.replace(folder / (name + '.png'))
        metrics_tmp = folder / (name + '.json.tmp')
        metrics_tmp.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        metrics_tmp.replace(folder / (name + '.json'))
        print(f"Validation step {step} | mean weighted loss: "
              f"app {metrics['baseline']['weighted_total']:.4f}, "
              f"candidate {metrics['candidate']['weighted_total']:.4f}", flush=True)
