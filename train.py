import argparse
from copy import deepcopy
from hashlib import file_digest, sha256
from math import ceil
from pathlib import Path
import re

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.models import APP_DECODER, Decoder, VGGEncoder
from utils.utils import (ImageFolderDataset, adaptive_instance_normalization, calc_mean_std,
                         get_transform, style_statistics_loss)
from utils.validation import Validation, load_pairs


ROOT = Path(__file__).resolve().parent
INPUT_PATHS = ('content_dir', 'style_dir', 'vgg', 'validation_dir')


def run_options(args):
    options = vars(args).copy()
    for key in ('experiment', 'resume', 'decoder_path', 'optimizer_path'):
        options.pop(key)
    for key in INPUT_PATHS:
        path = options[key].resolve()
        options[key] = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return options


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune an AdaIN decoder and compare it to the app.")
    parser.add_argument('--content_dir', type=Path, default=ROOT / 'content_data')
    parser.add_argument('--style_dir', type=Path, default=ROOT / 'style_data')
    parser.add_argument('--vgg', type=Path, default=ROOT / 'vgg_normalised.pth')
    parser.add_argument('--validation_dir', type=Path, default=ROOT / 'validation')
    parser.add_argument('--experiment', default='pilot', help="Name under experiment/, or absolute output directory")
    parser.add_argument('--content_size', type=int, default=512)
    parser.add_argument('--style_size', type=int, default=512)
    parser.add_argument('--final_size', type=int, default=256)
    parser.add_argument('--crop', action=argparse.BooleanOptionalAction, default=True,
                        help="Use random training crops (center crops with --no-crop)")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5, help="Starting LR for fine-tuning")
    parser.add_argument('--lr_decay', type=float, default=5e-5, help="Inverse decay per completed epoch")
    parser.add_argument('--epochs', type=int, default=1, help="Target number of completed content passes")
    parser.add_argument('--max_steps', type=int, help="Stop at this absolute optimizer step (0 clears a pilot limit)")
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--alpha', type=float, default=1.0, help="Fixed validation style strength")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=25, help="Update the loss display every N steps")
    parser.add_argument('--save_interval', '--save_every', type=int, default=200,
                        help="Checkpoint every N optimizer steps and at the requested stopping step")
    parser.add_argument('--resume', action='store_true', help="Continue from latest.pth in this experiment")
    parser.add_argument('--decoder_path', type=Path, help="Optional initial decoder, or a legacy epoch decoder")
    parser.add_argument('--optimizer_path', type=Path, help="Matching legacy epoch optimizer for migration")
    args = parser.parse_args()
    state = None
    if args.resume and not (args.optimizer_path or args.decoder_path):
        latest = ROOT / 'experiment' / args.experiment / 'latest.pth'
        if not latest.is_file():
            parser.error(f"No training checkpoint at {latest}")
        state = torch.load(latest, map_location='cpu', weights_only=True)
        if 'run_options' in state:
            defaults = state['run_options'].copy()
            for key in INPUT_PATHS:
                defaults[key] = ROOT / defaults[key]
            parser.set_defaults(**defaults)
            args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.save_interval, args.log_interval) < 1:
        parser.error("Batch size, epochs and intervals must be positive")
    if args.max_steps == 0:
        args.max_steps = None
    if args.max_steps is not None and args.max_steps < 0:
        parser.error("--max_steps must be nonnegative")
    if not 0 <= args.alpha <= 1 or args.lr <= 0 or min(args.lr_decay, args.content_weight, args.style_weight) < 0:
        parser.error("Check alpha [0,1], positive LR and nonnegative decay/loss weights")
    if args.final_size < 16 or args.final_size % 8:
        parser.error("--final_size must be at least 16 and divisible by 8 for this decoder")
    if args.optimizer_path and not (args.resume and args.decoder_path):
        parser.error("Legacy migration needs --resume, --decoder_path and --optimizer_path together")
    if args.resume and args.decoder_path and not args.optimizer_path:
        parser.error("For new-format checkpoints, use --resume alone. Legacy resume needs both weights files.")
    return args, state


class PairedEpoch(Dataset):
    """One shuffled pass over content, with independently reshuffled passes over style."""
    def __init__(self, content, style, seed, epoch, start_image):
        self.content = content
        self.style = style
        content_rng = torch.Generator().manual_seed(seed + 2 * epoch)
        style_rng = torch.Generator().manual_seed(seed + 2 * epoch + 1)
        content_order = torch.randperm(len(content), generator=content_rng).tolist()
        style_order = []
        while len(style_order) < len(content):
            style_order.extend(torch.randperm(len(style), generator=style_rng).tolist())
        self.pairs = list(zip(content_order, style_order))[start_image:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        content_id, style_id = self.pairs[index]
        return self.content[content_id], self.style[style_id]


def file_hash(path):
    with path.open('rb') as fp:
        return file_digest(fp, 'sha256').hexdigest()


def checkpoint_settings(args, content, style, held_out):
    # Relative dataset paths let a checkpoint move between Windows and Colab.
    settings = {key: getattr(args, key) for key in (
        'seed', 'batch_size', 'content_size', 'style_size', 'final_size', 'crop',
        'lr', 'lr_decay', 'content_weight', 'style_weight', 'alpha')}
    for name, dataset in [('content_index', content), ('style_index', style)]:
        settings[name] = sha256('\n'.join(dataset.files).encode()).hexdigest()
    settings['encoder'] = file_hash(args.vgg)
    settings['baseline'] = file_hash(APP_DECODER)
    settings['validation'] = [file_hash(p) for p in [args.validation_dir / 'pairs.json', *held_out]]
    return settings


def atomic_save(state, path):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def load_legacy(args, decoder, optimizer):
    match = re.fullmatch(r'decoder_epoch_(\d+)\.pth', args.decoder_path.name)
    if not match or args.optimizer_path.name != f'optimizer_epoch_{match[1]}.pth':
        raise ValueError("Expected matching decoder_epoch_N.pth and optimizer_epoch_N.pth files")
    decoder.load_weights(args.decoder_path)
    optimizer.load_state_dict(torch.load(args.optimizer_path, map_location='cpu', weights_only=True))
    steps = {int(value['step']) for value in optimizer.state.values()}
    if len(steps) != 1:
        raise ValueError("Cannot determine the legacy Adam step from optimizer state")
    print("Migrated legacy epoch boundary. The old run did not store its crop RNG.", flush=True)
    return int(match[1]), steps.pop()


def train_step(content, style, encoder, decoder, optimizer, weights):
    with torch.no_grad():
        content_features = encoder(content, is_test=True)
        style_features = encoder(style)
        target = adaptive_instance_normalization(content_features, style_features[-1])
        style_stats = [calc_mean_std(layer) for layer in style_features]
        del style_features
    features = encoder(decoder(target))
    content_loss = torch.nn.functional.mse_loss(features[-1], target) * weights[0]
    style_loss = style_statistics_loss(features, style_stats) * weights[1]
    loss = content_loss + style_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item(), content_loss.item(), style_loss.item()


def means_text(sums, count):
    return ', '.join(f'{label} {value/count:.4f}'
                     for label, value in zip(('total', 'content', 'style'), sums))


def main():
    args, state = parse_arguments()
    folder = ROOT / 'experiment' / args.experiment
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / 'latest.pth'
    if latest.exists() and (not args.resume or args.optimizer_path):
        raise ValueError(f"{latest} already exists. Use --resume or a fresh experiment name.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pairs = load_pairs(args.validation_dir)
    held_out = [path for _, photo, painting in pairs for path in (photo, painting)]
    content = ImageFolderDataset(args.content_dir, get_transform(args.content_size, args.crop, args.final_size), held_out)
    style = ImageFolderDataset(args.style_dir, get_transform(args.style_size, args.crop, args.final_size), held_out)
    batches = ceil(len(content) / args.batch_size)
    print(f"{device} | {len(content)} training photos, {len(style)} training paintings, "
          f"{batches} batches per full epoch | {folder}", flush=True)
    settings = checkpoint_settings(args, content, style, held_out)
    encoder = VGGEncoder(args.vgg).to(device).eval()
    baseline = Decoder().to(device).eval()
    baseline.load_weights(APP_DECODER)
    decoder = deepcopy(baseline).train()
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    epoch, offset, step = 0, 0, 0  # Completed epochs, batches in this epoch, total optimizer updates.
    sums = [0.0, 0.0, 0.0]
    if args.optimizer_path:
        epoch, step = load_legacy(args, decoder, optimizer)
    elif args.decoder_path:
        decoder.load_weights(args.decoder_path)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda n: 1 / (1 + args.lr_decay * n), last_epoch=epoch-1)
    if state is not None:
        if state['settings'] != settings:
            changed = [k for k in settings if settings[k] != state['settings'].get(k)]
            raise ValueError(f"Resume settings differ: {', '.join(changed)}. Use the original run settings.")
        decoder.load_state_dict(state['decoder'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch, offset, step = state['completed_epochs'], state['step_in_epoch'], state['global_step']
        sums = state['loss_sums']
        torch.set_rng_state(state['crop_rng'])
        if device.type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'])
        del state
    else:
        torch.manual_seed(args.seed)
        (folder / 'args.txt').write_text('\n'.join(f'{k}: {v}' for k, v in vars(args).items()), encoding='utf-8')

    weights = (args.content_weight, args.style_weight)
    validation = Validation(pairs, encoder, baseline, device, args.alpha, weights)
    del baseline

    def write_outputs():
        atomic_save(decoder.state_dict(), folder / f'decoder_step_{step:06d}.pth')
        validation.write(decoder, folder, step)

    # A disconnect during preview rendering still leaves a usable training checkpoint.
    if latest.exists():
        if not all((folder / name).exists() for name in (
            f'decoder_step_{step:06d}.pth', f'comparison_{step:06d}.png', f'comparison_{step:06d}.json')):
            write_outputs()
        print(f"Resumed at step {step}, epoch {epoch+1}, {offset}/{batches} batches complete.", flush=True)

    while epoch < args.epochs and (args.max_steps is None or step < args.max_steps):
        dataset = PairedEpoch(content, style, args.seed, epoch, offset * args.batch_size)
        # A dedicated loader RNG keeps iterator creation from consuming the saved crop RNG.
        loader = DataLoader(dataset, batch_size=args.batch_size, generator=torch.Generator().manual_seed(0),
                            pin_memory=device.type == 'cuda')
        with tqdm(total=batches, initial=offset, desc=f'Epoch {epoch+1}', unit='batch') as progress:
            for photo, painting in loader:
                losses = train_step(photo.to(device), painting.to(device), encoder, decoder, optimizer, weights)
                step += 1
                offset += 1
                sums = [total + value for total, value in zip(sums, losses)]
                progress.update()
                if step % args.log_interval == 0:
                    progress.set_postfix_str(f'step {step}, mean {means_text(sums, offset)}')
                if offset == batches:
                    tqdm.write(f'Epoch {epoch+1} | {offset} batches | step {step} | mean {means_text(sums, offset)}')
                    scheduler.step()
                    epoch += 1
                    offset, sums = 0, [0.0, 0.0, 0.0]
                finished = epoch >= args.epochs or (args.max_steps is not None and step >= args.max_steps)
                if step % args.save_interval == 0 or finished:
                    atomic_save(dict(decoder=decoder.state_dict(), optimizer=optimizer.state_dict(),
                                     scheduler=scheduler.state_dict(), completed_epochs=epoch, step_in_epoch=offset,
                                     global_step=step, loss_sums=sums, crop_rng=torch.get_rng_state(),
                                     cuda_rng=torch.cuda.get_rng_state() if device.type == 'cuda' else None,
                                     settings=settings, run_options=run_options(args)), latest)
                    write_outputs()
                    if offset:
                        tqdm.write(f'Epoch {epoch+1} | {offset}/{batches} batches | step {step} | mean {means_text(sums, offset)}')
                if finished:
                    return


if __name__ == '__main__':
    main()
