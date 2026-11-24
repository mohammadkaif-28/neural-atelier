# AdaIN fine-tuning pilot

Run these commands from the project folder. Keep `vgg_normalised.pth`,
`experiment/final_exp/decoder_final.pth` and the bundled `validation/` folder
in their project locations. Training uses PyTorch, a matching torchvision,
Pillow 10.1 or newer, and tqdm. The Flask dependencies are only needed for the app.

## Start a Colab pilot

Mount Google Drive and select a GPU runtime. The example assumes the extracted
photo and painting directories are under `/content`. Adjust those two paths to
match your datasets. The absolute experiment path saves progress on Drive.

```bash
python train.py \
  --content_dir /content/test2017 \
  --style_dir /content/train_1 \
  --experiment /content/drive/MyDrive/NST_runs/pilot \
  --batch_size 4 --max_steps 600 --save_interval 200
```

In a Colab notebook, prefix the command with `!` or start the cell with `%%bash`.
Choose the image directories that will also be available when you resume. The
trainer reports the number of eligible files before its first update. With the
full local datasets and the bundled holdouts this gives 40,667 photos, 11,022
paintings and 10,167 batches at batch size 4, including the final partial batch.

Fine-tuning starts from the decoder used by the app, with a learning rate of
`1e-5`. Training resizes the shorter side of each image to 512 pixels and
samples a 256-pixel square. Each epoch visits every eligible content photo once.
Paintings reshuffle each time their pool is exhausted. Reported training losses
are means over processed batches, and the LR scheduler advances after each full epoch.

## Resume

```bash
python train.py --experiment /content/drive/MyDrive/NST_runs/pilot --resume
```

The run restores its options and continues from the last saved optimizer step.
If the data moved, pass the new `--content_dir` and `--style_dir` with the
same image files and directory structure inside each root. To extend a finished
pilot, pass a larger absolute `--max_steps`. To continue through full epochs,
pass `--max_steps 0 --epochs 2` (two total completed epochs).

## Compare checkpoints

Open `comparison_NNNNNN.png` in the experiment folder. Each row shows the
photo, reference painting, original app output, and fine-tuned output at that
step. Both decoders use the app's 512-pixel shorter-side resize and the same
alpha (default 1.0). Compare facial features and glasses on the portrait,
mountain outlines and sky texture on the landscape, and window edges and
paint texture on the building.

`comparison_NNNNNN.json` gives per-pair content and style losses and their
three-pair means. The weighted total uses the training weights (defaults 1 and
10). A smaller total can coincide with less detail in an individual photo.
The six validation files and byte-identical copies are excluded from both training pools.

`latest.pth` is the rolling resumable checkpoint. Each saved comparison also
has a standalone `decoder_step_NNNNNN.pth` for keeping that candidate.
The app loads `experiment/final_exp/decoder_final.pth` for the original
single-alpha path. The portrait preset and manual controls use
`experiment/pilot_600/decoder_step_000200.pth`. See [README.md](README.md)
for the inference setup and a visual comparison.
