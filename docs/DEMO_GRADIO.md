# Gradio Demo (Local Only)

## Install
Use the project environment and install demo extras:

```powershell
python -m pip install -e ".[dev,demo]"
```

## Run
```powershell
python -m faceanalyze2.demo.gradio_app
```

Optional host/port:
```powershell
python -m faceanalyze2.demo.gradio_app --host 127.0.0.1 --port 7860
```

## PHI / Security Notice
- This demo may display patient frames and derived artifacts.
- External sharing is prohibited.
- `share=True` is intentionally disabled. The app launches for local use only.

## Demo Flow
1. Upload an `.mp4` file.
2. Select motion: `big-smile`, `blinking-motion`, or `eyebrow-motion`.
3. Keep **Ensure pipeline run** ON for end-to-end preparation.
4. Click **Run**.
5. Review:
   - 5 key images
   - metrics table
   - optional ROI gallery
   - optional viewer HTML and timeseries CSV downloads
   - raw JSON payload

## Notes
- Uploaded files are copied into `demo_inputs/` with UUID prefixes.
- Generated helper files (e.g., decoded CSV) can be created in `demo_outputs/`.
- Both folders are ignored by git.

