# Docling OCR GPU API - Deployment Guide

This guide explains how to deploy the Docling OCR service alongside your existing embedding model on the same GPU VM, using Cloudflare for external access.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              YOUR SETUP                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                                              ┌─────────────────────────────┐│
│                                              │   GPU VM (caplyst.live)     ││
│   ┌─────────────────────┐                    │                             ││
│   │     RAGFlow         │    PDF bytes       │  ┌─────────────────────┐   ││
│   │  (Your Server)      │ ──────────────────>│  │  OCR Server :8001   │   ││
│   │                     │   via Cloudflare   │  │  - Preprocessing    │   ││
│   │                     │<────────────────── │  │  - Docling OCR      │   ││
│   │                     │   pages[], tables  │  │  - TableFormer      │   ││
│   │                     │                    │  └─────────────────────┘   ││
│   │                     │    text chunks     │                             ││
│   │                     │ ──────────────────>│  ┌─────────────────────┐   ││
│   │                     │   via Cloudflare   │  │  Embed Server :8000 │   ││
│   │                     │<────────────────── │  │  nomic-embed-text   │   ││
│   └─────────────────────┘   vectors[]        │  └─────────────────────┘   ││
│                                              │                             ││
│                                              └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘

Services on Same GPU VM:
  - Embedding API: https://embed.caplyst.live/v1/embeddings  (port 8000)
  - OCR API:       https://ocr.caplyst.live/v1/ocr/parse-pages  (port 8001)
```

---

## Quick Start (Same VM as Embedding Model)

Since you already have the embedding model running on your GPU VM with Cloudflare, here's the quick setup to add OCR:

### Step 1: Copy Docling to GPU VM

```bash
# From your local machine, copy the Docling folder to your GPU VM
scp -r /home/feras/ragflow/Docling/Docling user@gpu-vm:/path/to/services/

# Or if using git, push to your repo and pull on the VM
git add Docling/
git commit -m "Add Docling OCR server"
git push

# Then on GPU VM:
git pull
```

### Step 2: Install Dependencies on GPU VM

```bash
# SSH into your GPU VM
ssh user@gpu-vm

# Navigate to Docling folder
cd /path/to/services/Docling

# Use existing venv or create new one
# Option A: Use same venv as embedding (if compatible)
source /path/to/embed/venv/bin/activate

# Option B: Create separate venv for OCR
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install wordninja  # Important for OCR text spacing fix!

# Verify installation
python -c "import docling; print('docling OK')"
python -c "import wordninja; print('wordninja OK')"
```

### Step 3: Start the OCR Server

```bash
# Test run (foreground)
uvicorn server:app --host 0.0.0.0 --port 8001

# Or run in background
nohup uvicorn server:app --host 0.0.0.0 --port 8001 > ocr.log 2>&1 &

# Verify it's running
curl http://localhost:8001/health
# Should return: {"status": "healthy", "service": "docling-ocr"}
```

### Step 4: Configure Cloudflare Tunnel

Add a new public hostname in your Cloudflare Zero Trust dashboard:

1. Go to **Cloudflare Zero Trust** > **Access** > **Tunnels**
2. Select your existing tunnel (the one used for embed.caplyst.live)
3. Click **Configure** > **Public Hostname**
4. Add new hostname:
   - **Subdomain**: `ocr`
   - **Domain**: `caplyst.live`
   - **Service**: `http://localhost:8001`
5. Save

After a few minutes, test:
```bash
curl https://ocr.caplyst.live/health
# Should return: {"status": "healthy", "service": "docling-ocr"}
```

### Step 5: Configure RAGFlow

On your RAGFlow server:

```bash
# Set the OCR API URL
export DOCLING_OCR_API_URL="https://ocr.caplyst.live"

# Add to .bashrc for persistence
echo 'export DOCLING_OCR_API_URL="https://ocr.caplyst.live"' >> ~/.bashrc

# Restart RAGFlow to pick up the new environment variable
```

### Step 6: Test End-to-End

```bash
# Test from RAGFlow server
curl https://ocr.caplyst.live/health

# Upload a PDF in RAGFlow UI and check logs for:
# "Using Docling GPU API at https://ocr.caplyst.live..."
```

---

## Running as a Systemd Service (Production)

For production, run the OCR server as a systemd service:

```bash
sudo nano /etc/systemd/system/docling-ocr.service
```

Add:
```ini
[Unit]
Description=Docling OCR FastAPI Server
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=/path/to/services/Docling
Environment="PATH=/path/to/services/Docling/venv/bin"
ExecStart=/path/to/services/Docling/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable docling-ocr
sudo systemctl start docling-ocr
sudo systemctl status docling-ocr
```

---

## Files on GPU VM

After copying, your GPU VM should have:

```
/path/to/services/Docling/
├── server.py           # FastAPI OCR server (main entry point)
├── preprocessor.py     # Image preprocessing (deskew, denoise)
├── extractor.py        # Docling extraction logic
├── postprocessor.py    # Text spacing correction (wordninja)
├── requirements.txt    # Python dependencies
├── DEPLOYMENT_GUIDE.md # This guide
└── venv/               # Virtual environment (created by you)
```

---

## Files on RAGFlow

These files were already modified in your RAGFlow installation:

```
/home/feras/ragflow/
├── deepdoc/parser/docling/
│   ├── client.py       # API client (calls GPU server)
│   └── parser.py       # DoclingParser (uses client)
└── rag/app/
    └── naive.py        # Entry point (reads DOCLING_OCR_API_URL)
```

---

## API Endpoints

### Health Check
```bash
curl https://ocr.caplyst.live/health
# Returns: {"status": "healthy", "service": "docling-ocr"}
```

### Parse PDF (Full Response)
```bash
curl -X POST https://ocr.caplyst.live/v1/ocr/parse \
  -H "Content-Type: application/json" \
  -d '{"pdf_base64": "<base64-pdf>", "enable_preprocessing": true}'
```

### Parse PDF (Simple - Used by RAGFlow)
```bash
curl -X POST https://ocr.caplyst.live/v1/ocr/parse-pages \
  -H "Content-Type: application/json" \
  -d '{"pdf_base64": "<base64-pdf>"}'
```

---

## Configuration Options

### Environment Variable (RAGFlow)
```bash
export DOCLING_OCR_API_URL="https://ocr.caplyst.live"
```

### Request Options (sent with each PDF)

| Option | Default | Description |
|--------|---------|-------------|
| `enable_preprocessing` | `true` | Run image preprocessing |
| `enable_spacing_fix` | `true` | Fix merged words (wordninja) |
| `enable_deskew` | `true` | Straighten rotated pages |
| `enable_denoising` | `true` | Apply CLAHE + bilateral filter |
| `target_dpi` | `600` | DPI for image processing |
| `ocr_lang` | `["en"]` | OCR languages |

---

## Troubleshooting

### "Connection refused" error
```bash
# Check if server is running on GPU VM
ssh user@gpu-vm
curl http://localhost:8001/health

# Check systemd service
sudo systemctl status docling-ocr
sudo journalctl -u docling-ocr -f
```

### "502 Bad Gateway" from Cloudflare
- Server not running on port 8001
- Cloudflare tunnel not configured correctly
- Check tunnel logs in Cloudflare dashboard

### "Timeout" error
- Large PDFs take time (50 pages ~ 3-5 min)
- Check GPU memory: `nvidia-smi`
- Check server logs for errors

### Poor OCR quality
- Ensure `target_dpi: 600`
- Enable all preprocessing options
- Check if `wordninja` is installed

---

## Monitoring

### Check GPU Usage
```bash
nvidia-smi -l 1
```

### Check Server Logs
```bash
# If systemd
sudo journalctl -u docling-ocr -f

# If running directly
tail -f ocr.log
```

### Check RAGFlow Logs
Look for:
```
Using Docling GPU API at https://ocr.caplyst.live...
Docling GPU API extracted X pages.
```

---

## Quick Reference

```bash
# Start OCR server manually
cd /path/to/Docling && source venv/bin/activate && uvicorn server:app --host 0.0.0.0 --port 8001

# Check service status
sudo systemctl status docling-ocr

# View logs
sudo journalctl -u docling-ocr -f

# Test health
curl https://ocr.caplyst.live/health

# Test from RAGFlow
export DOCLING_OCR_API_URL="https://ocr.caplyst.live"
```

---

## Summary

You now have two services running on the same GPU VM:

| Service | Port | Cloudflare URL | Purpose |
|---------|------|----------------|---------|
| Embedding | 8000 | https://embed.caplyst.live | Text → Vectors |
| OCR | 8001 | https://ocr.caplyst.live | PDF → Text + Tables |

Both services share the GPU and are accessed via Cloudflare tunnels for security and easy access from RAGFlow.
