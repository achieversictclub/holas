HOLAS v2 — Deploy & Secure Guide
--------------------------------

1) Prepare environment (local or Render / Hugging Face Spaces)
   - Python 3.10+
   - GPU recommended for image generation (stable-diffusion)

2) Files to place in repo:
   - holas_v2.py
   - requirements.txt
   - (optional) holas_brand.yaml (colors, theme)

3) Secrets / environment variables (important)
   - HF_TOKEN: Hugging Face token (if using gated HF models or diffusers)
   - MODEL_NAME: Hugging Face model id (optional)
   - DIFFUSION_MODEL: e.g. runwayml/stable-diffusion-v1-5
   - PIN: admin PIN (default 1234; override in production)
   - BASE_URL: optional public URL where app is hosted (to create image links)
   - WP_URL, WP_USER, WP_APP_PASSWORD: (optional) for real WordPress publishing
   - ALLOW_REAL_PUBLISH: set to "true" to enable real publishing flow

4) Run locally (test):
   - pip install -r requirements.txt
   - export HF_TOKEN=... (if needed)
   - python holas_v2.py
   - Open browser at http://127.0.0.1:7860

5) Deploy on Render / Hugging Face Spaces:
   - Push repo to GitHub
   - Create Web Service (Render) or New Space (Hugging Face)
   - Provide environment vars / secrets through platform UI
   - Deploy and visit the public URL

6) Using WordPress publishing:
   - Create an application password for WP user (WP Admin → Users → Profile → Application Passwords)
   - Put WP_URL, WP_USER, WP_APP_PASSWORD into environment variables
   - Set ALLOW_REAL_PUBLISH=true to enable real publish
   - Use the Publish panel in HOLAS to publish. You must confirm by typing "PUBLISH NOW" and enter PIN.

Security & ethics:
- Never commit HF_TOKEN or WP credentials to git.
- For production, use a secret manager (Render secrets, HF Spaces secrets, or Vault).
- HOLAS will refuse and explain if asked for hacking assistance. It provides safe alternatives.

