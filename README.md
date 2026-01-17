# Logo Post-Processing Pipeline

AI-powered logo processing pipeline that transforms raster logos into clean, scalable vector formats.

## Pipeline Steps

1. **Upscale** - 4x upscaling using Real-ESRGAN
2. **Background Removal** - AI-based background removal (BiRefNet-HR/RMBG)
3. **Cleanup** - Trim transparent areas and quantize colors
4. **Vectorize** - Convert to SVG using vtracer (color and B&W variants)
5. **Optimize** - SVG optimization with scour
6. **Export** - Generate PDF and EPS formats

## Usage

### CLI

```bash
python main.py path/to/logo.png
```

Options:
- `-o, --output-dir` - Output directory (default: `data/outputs`)
- `--max-colors` - Max colors for quantization (default: 12)
- `--trim-pad` - Padding after trim (default: 24)

### API

Deploy to Modal:

```bash
modal deploy modal_app.py
```

The API exposes a `/process-logo` endpoint that accepts a logo file and email address, processes the logo, and sends results via email.

## Environment Variables

Create a `.env` file:

```
MODAL_APP_NAME=your-app-name
RESEND_API_KEY=your-resend-api-key
EMAIL_FROM=noreply@yourdomain.com
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
src/
  api/          # FastAPI server
  models/       # Background removal models (BiRefNet, RMBG, SAM2, etc.)
  processors/   # Image processing (upscaler, vectorizer, cleanup, exporter)
  config.py     # Configuration
main.py         # CLI entry point
modal_app.py    # Modal deployment
```
