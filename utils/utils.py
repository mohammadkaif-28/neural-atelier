from hashlib import sha256
from pathlib import Path
import warnings

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None, held_out=()):
        self.root = Path(root)
        self.transform = transform
        self.skipped_images = set()
        if not self.root.is_dir():
            raise ValueError(f"Image directory does not exist: {self.root}")
        # Only hash candidates with a matching size. Renamed holdout copies are excluded too.
        sizes = {p.stat().st_size for p in held_out}
        digests = {sha256(p.read_bytes()).digest() for p in held_out}
        self.files = []
        for path in self.root.rglob('*'):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'} or not path.is_file():
                continue
            if path.stat().st_size in sizes and sha256(path.read_bytes()).digest() in digests:
                continue
            self.files.append(path.relative_to(self.root).as_posix())
        self.files.sort()
        if not self.files:
            raise ValueError(f"No training images left in {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Preserve checkpoint indices. Substitute the next eligible file in this same pool.
        for offset in range(len(self.files)):
            name = self.files[(idx + offset) % len(self.files)]
            if name in self.skipped_images:
                continue
            path = self.root / name
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', Image.DecompressionBombWarning)
                    with Image.open(path) as source:
                        image = source.convert('RGB')
            except (OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
                if isinstance(exc, OSError) and exc.errno is not None:
                    raise  # Filesystem failures need attention, unlike bad image data.
                self.skipped_images.add(name)
                print(f"Skipping image {path}: {exc}", flush=True)
                continue
            return self.transform(image) if self.transform else image
        raise RuntimeError(f"No decodable training images left in {self.root}")
    
   

def get_transform(size, crop, final_size):
    if size < final_size or final_size <= 0:
        raise ValueError("The shorter-side resize must be at least the positive crop size")
    crop_transform = transforms.RandomCrop if crop else transforms.CenterCrop
    return transforms.Compose([
        transforms.Resize(size),
        crop_transform(final_size),
        transforms.ToTensor(),
    ])


def inference_tensor(image, device):
    transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
    return transform(image).unsqueeze(0).to(device)


def blend_features(content, style, alpha):
    target = adaptive_instance_normalization(content, style)
    return alpha * target + (1 - alpha) * content


def face_mask(original):
    import cv2
    import numpy as np

    if not hasattr(cv2, 'CascadeClassifier'):
        raise RuntimeError("Portrait mode requires OpenCV 4. Start the app with Start_NST.cmd.")
    gray = transforms.functional.rgb_to_grayscale(original)[0, 0].mul(255).byte().cpu().numpy()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.equalizeHist(gray)
    mask = np.zeros(gray.shape, np.float32)
    boxes = list(detector.detectMultiScale(gray, 1.1, 5, minSize=(32, 32)))
    if not boxes:
        profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        boxes.extend(profile.detectMultiScale(gray, 1.1, 5, minSize=(32, 32)))
        for x, y, w, h in profile.detectMultiScale(cv2.flip(gray, 1), 1.1, 5, minSize=(32, 32)):
            boxes.append((gray.shape[1]-x-w, y, w, h))
    for x, y, w, h in boxes:
        region = np.zeros_like(mask)
        cv2.ellipse(region, (x+w//2, y+int(.53*h)), (int(.46*w), int(.54*h)), 0, 0, 360, 1, -1)
        mask = np.maximum(mask, cv2.GaussianBlur(region, (0, 0), max(2, .045*w)))
    return torch.from_numpy(mask).to(original)[None, None]


def portrait_output(original, styled, reference, face_detail=0.70):
    gray = transforms.functional.rgb_to_grayscale
    blur = transforms.functional.gaussian_blur
    photo_luma = gray(original)
    ref_luma = gray(reference)
    if (reference - ref_luma).abs().mean() < 0.035 and ref_luma.mean() > 0.5:
        # Graphite on paper: dodge shading gives light paper and dark contours.
        paper = (photo_luma / blur(photo_luma, [31, 31], [6., 6.]).clamp_min(.01)).clamp(0, 1)
        pencil = .62 * paper + .38 * gray(styled).clamp(0, 1)
        # Match the reference's light and dark tones.
        sample = ref_luma[..., ::4, ::4]
        dark, light = torch.quantile(sample.flatten(), sample.new_tensor([.05, .95]))
        lo, hi = torch.quantile(pencil[..., ::4, ::4].flatten(), pencil.new_tensor([.05, .95]))
        styled = (dark + (pencil-lo) * (light-dark) / (hi-lo).clamp_min(.05)).clamp(0, 1).expand_as(original)
    # Restore facial luminance detail without bringing back the photo's colors.
    if face_detail == 0:
        return styled
    face = face_mask(original)
    out_luma = gray(styled)
    original_detail = photo_luma - blur(photo_luma, [21, 21], [4., 4.])
    styled_detail = out_luma - blur(out_luma, [21, 21], [4., 4.])
    return styled + face_detail * face * (original_detail - styled_detail)


@torch.no_grad()
def style_transfer(content_image, style_image, encoder, decoder, alpha, device, *, preserve_detail=False, face_detail=0.70):
    original = inference_tensor(content_image, device)
    if preserve_detail and alpha == 0:
        return original
    content = encoder(original, is_test=True)
    reference = inference_tensor(style_image, device)
    style = encoder(reference, is_test=True)
    styled = decoder(blend_features(content, style, alpha))
    if preserve_detail:
        h, w = original.shape[-2:]
        output = portrait_output(original, styled[..., :h, :w], reference, face_detail)
        # Fade the pencil treatment and decoder reconstruction down to the original.
        if alpha < 0.8:
            return torch.lerp(original, output, alpha / 0.8)
        return output
    return styled


def style_statistics_loss(features, target_stats):
    loss = 0
    for layer, (style_mean, style_std) in zip(features, target_stats):
        mean, std = calc_mean_std(layer)
        loss = loss + torch.nn.functional.mse_loss(mean, style_mean)
        loss = loss + torch.nn.functional.mse_loss(std, style_std)
    return loss

def adaptive_instance_normalization(content_features, style_features):
    # [batch size, channels, h, w]
    size = content_features.size()
    style_mean, style_std = calc_mean_std(style_features)
    content_mean, content_std = calc_mean_std(content_features)
    normalized_content_feat = (content_features - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5):
    # [batch size, channels, h, w]
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std

