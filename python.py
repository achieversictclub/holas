#!/usr/bin/env python3
"""
HOLAS v2 — Open-minded Luxury Personal Assistant
Features:
- Open-minded persona and configurable tone (activated after admin PIN)
- Chat (open LLM via Hugging Face, fallback to gpt2)
- Image generation via Diffusers (if HF_TOKEN present)
- Admin panel: change PIN, view audit, switch models
- Website management (WordPress REST API) with safe dry-run and optional real publish
- Generates real links for images/posts when BASE_URL or WP is configured
- Audit trail (audit_log.json) for all high-risk actions
- Designed for deployment (Render, Hugging Face Spaces, local)
"""

import os
import json
import uuid
import time
import logging
from datetime import datetime
from pathlib import Path
from io import BytesIO
from base64 import b64decode

import gradio as gr
import requests
import torch
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Optional diffusers import - fallback handled
try:
    from diffusers import StableDiffusionPipeline
except Exception:
    StableDiffusionPipeline = None

# ---------- Configuration (use environment variables) ----------
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token for gated models / diffusers
DEFAULT_MODEL = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-instruct")
DIFFUSION_MODEL = os.getenv("DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5")
PIN_ENV = os.getenv("PIN", "1234")  # default PIN - override in secrets
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "media"))
AUDIT_FILE = Path("audit_log.json")
CONFIG_FILE = Path("holas_config.json")
BASE_URL = os.getenv("BASE_URL", "")  # public base URL where app is hosted (optional)
WP_URL = os.getenv("WP_URL")  # e.g. "https://example.com"
WP_USER = os.getenv("WP_USER")
WP_APP_PASSWORD = os.getenv("WP_APP_PASSWORD")  # WordPress application password (recommended)
ALLOW_REAL_PUBLISH = os.getenv("ALLOW_REAL_PUBLISH", "false").lower() in ("1","true","yes")

# Ensure dirs
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("holas_v2")

# ---------- Audit helpers ----------
def read_audit():
    if AUDIT_FILE.exists():
        try:
            return json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def write_audit(entry):
    logs = read_audit()
    logs.append(entry)
    AUDIT_FILE.write_text(json.dumps(logs, indent=2), encoding="utf-8")

def audit(actor, action, details=None):
    entry = {
        "id": str(uuid.uuid4()),
        "time": datetime.utcnow().isoformat() + "Z",
        "actor": actor,
        "action": action,
        "details": details or {}
    }
    logger.info("AUDIT: %s", entry)
    write_audit(entry)
    return entry

# ---------- Config persistence (PIN / theme / model) ----------
DEFAULT_CONFIG = {"pin": PIN_ENV, "theme": "blue_gold", "model": DEFAULT_MODEL}
if not CONFIG_FILE.exists():
    CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")

def read_config():
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_CONFIG.copy()

def write_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    audit("admin", "config_updated", cfg)

# ---------- LLM loading (open models fallback) ----------
def safe_load_llm(model_name):
    try:
        logger.info("Loading LLM: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None
        )
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        logger.info("LLM loaded.")
        return gen
    except Exception as e:
        logger.warning("Failed to load %s: %s. Falling back to gpt2.", model_name, str(e))
        return pipeline("text-generation", model="gpt2")

CFG = read_config()
LLM_PIPE = safe_load_llm(CFG.get("model", DEFAULT_MODEL))

def switch_model(new_model):
    global LLM_PIPE
    audit("admin", "switch_model", {"from": CFG.get("model"), "to": new_model})
    try:
        LLM_PIPE = safe_load_llm(new_model)
        cfg = read_config()
        cfg["model"] = new_model
        write_config(cfg)
        return True, f"Model switched to {new_model}"
    except Exception as e:
        return False, f"Failed to switch model: {e}"

# ---------- Diffusion loading (image generation) ----------
SD_PIPE = None
def load_sd_if_available(model_name):
    global SD_PIPE
    if StableDiffusionPipeline is None:
        logger.warning("diffusers not available in environment.")
        SD_PIPE = None
        return
    try:
        if HF_TOKEN is None:
            logger.warning("HF_TOKEN not set; some models may require it.")
        logger.info("Loading SD model: %s", model_name)
        SD_PIPE = StableDiffusionPipeline.from_pretrained(model_name, use_auth_token=HF_TOKEN)
        if torch.cuda.is_available():
            SD_PIPE = SD_PIPE.to("cuda")
        logger.info("SD pipeline ready.")
    except Exception as e:
        logger.error("Failed to load SD model: %s", e)
        SD_PIPE = None

load_sd_if_available(DIFFUSION_MODEL)

# ---------- Personality conditioning (open-minded persona) ----------
# A small wrapper that conditions generation with a persona prompt.
PERSONA_INSTRUCTION = (
    "You are HOLAS, an open-minded, creative, respectful, and helpful assistant. "
    "Be confident, concise, and show empathy. Avoid providing illicit or harmful instructions. "
    "When localizing for Uganda, be mindful of cultural references and be uplifting."
)

def generate_text(prompt, max_tokens=250):
    audit("user", "text_request", {"prompt_snippet": prompt[:300]})
    full_prompt = PERSONA_INSTRUCTION + "\n\nUser: " + prompt + "\nHOLAS:"
    try:
        out = LLM_PIPE(full_prompt, max_length=max_tokens, do_sample=True, temperature=0.8, top_p=0.9)
        # pipeline returns list
        if isinstance(out, list) and "generated_text" in out[0]:
            text = out[0]["generated_text"]
            # remove prompt prefix if present
            text = text.split("HOLAS:")[-1].strip()
        else:
            text = str(out)
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        text = "Sorry — I'm currently unable to generate a full response."
    audit("holas", "text_generated", {"prompt_snippet": prompt[:200], "result_snippet": text[:200]})
    return text

# ---------- Image generation (returns PIL Image and local path) ----------
def generate_image(prompt, steps=28):
    if SD_PIPE is None:
        return None, None, "Image model not available in this deployment."
    audit("user", "image_request", {"prompt": prompt[:300]})
    try:
        res = SD_PIPE(prompt, num_inference_steps=int(steps))
        image = res.images[0]
        fname = MEDIA_DIR / f"holas_img_{int(time.time())}.png"
        image.save(fname)
        audit("holas", "image_generated", {"prompt_snippet": prompt[:200], "file": str(fname)})
        # Build public link if BASE_URL is present
        public_link = ""
        if BASE_URL:
            # convention: images served under BASE_URL/media/<filename>
            public_link = BASE_URL.rstrip("/") + f"/{str(fname).replace('\\','/')}"
        return image, str(fname), public_link
    except Exception as e:
        logger.error("Image generation error: %s", e)
        return None, None, str(e)

# ---------- WordPress publish helpers (safe) ----------
def wp_publish_post(title, content, image_local_path=None):
    """
    Attempt to publish to WordPress via REST API using WP_USER and WP_APP_PASSWORD.
    Requires WP_URL, WP_USER, WP_APP_PASSWORD env vars to be set.
    Returns dict with success status and details.
    """
    audit("holas", "wp_publish_request", {"title": title})
    if not (WP_URL and WP_USER and WP_APP_PASSWORD and ALLOW_REAL_PUBLISH):
        logger.info("Real publishing is disabled or missing credentials. Returning dry-run preview.")
        preview = {
            "title": title,
            "content_snippet": content[:400],
            "status": "dry-run",
            "issue": "Real publishing not enabled or credentials not provided."
        }
        return {"success": False, "preview": preview}

    try:
        # Optional: upload image to WP media endpoint first
        headers = {}
        auth = (WP_USER, WP_APP_PASSWORD)
        media_url = None
        if image_local_path:
            # Upload media
            files = {
                'file': open(image_local_path, 'rb')
            }
            media_endpoint = WP_URL.rstrip("/") + "/wp-json/wp/v2/media"
            r = requests.post(media_endpoint, auth=auth, files=files)
            if r.status_code in (200,201):
                media_resp = r.json()
                media_url = media_resp.get("source_url")
                audit("holas", "wp_media_uploaded", {"media_url": media_url})
            else:
                audit("holas", "wp_media_upload_failed", {"status": r.status_code, "body": r.text})
                logger.warning("Media upload failed: %s", r.text)

        post_endpoint = WP_URL.rstrip("/") + "/wp-json/wp/v2/posts"
        post_data = {
            "title": title,
            "content": content,
            "status": "publish"
        }
        if media_url:
            post_data["featured_media"] = media_resp.get("id")
        r = requests.post(post_endpoint, auth=auth, json=post_data)
        if r.status_code in (200,201):
            resp = r.json()
            audit("holas", "wp_published", {"post_id": resp.get("id"), "link": resp.get("link")})
            return {"success": True, "post": resp}
        else:
            audit("holas", "wp_publish_failed", {"status": r.status_code, "body": r.text})
            return {"success": False, "error": r.text, "status": r.status_code}
    except Exception as e:
        audit("holas", "wp_publish_exception", {"error": str(e)})
        return {"success": False, "error": str(e)}

# ---------- Helper: build public URL for local file if BASE_URL provided ----------
def build_public_link(local_path):
    if not local_path:
        return ""
    if BASE_URL:
        return BASE_URL.rstrip("/") + "/" + str(Path(local_path).as_posix())
    return ""

# ---------- Simulated safe publish (dry-run) ----------
def dry_run_publish(title, content, image_local_path=None):
    audit("user", "dry_run_publish", {"title": title})
    preview = {
        "title": title,
        "content_snippet": content[:400],
        "image_local": image_local_path,
        "image_public": build_public_link(image_local_path),
        "status": "draft",
        "estimated_downtime": "none",
        "preview_time": datetime.utcnow().isoformat()+"Z"
    }
    return preview

# ---------- Gradio UI assembly ----------
def build_ui():
    # CSS theme
    theme_css = """
    <style>
      body { background: linear-gradient(180deg,#071030,#000); color: #eaeaea; }
      .holas-header { display:flex; align-items:center; gap:12px; }
      .holas-title { color:#FFD700; font-weight:700; font-size:28px; }
      .holas-sub { color:#0B2D8D; font-weight:600; margin-left:6px; }
      .card { background: rgba(255,255,255,0.03); padding:12px; border-radius:10px; }
    </style>
    """

    with gr.Blocks(css="", title="HOLAS — Achievers ICT Club") as demo:
        gr.HTML(theme_css)
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<div class='holas-header'><div class='holas-title'>HOLAS</div><div class='holas-sub'>— Achievers ICT Club</div></div>")
                gr.Markdown("**Luxury Intelligence · Personal Control**")
                gr.Markdown("---")
                # PIN unlock controls
                pin_in = gr.Textbox(label="Enter admin PIN", type="password", placeholder="PIN")
                unlock_btn = gr.Button("Unlock")
                unlock_status = gr.Textbox(label="Status", interactive=False, value="Locked — enter PIN")
                # Startup area for animation and chime (displayed after unlock)
                startup_html = gr.HTML(STARTUP_ANIMATION_HTML(), visible=False)
                startup_audio = gr.Audio(value=None, visible=False)

                # Visible main content (tabs) hidden until unlock success
                tabs = gr.Tabs(visible=False)

            with gr.Column(scale=2):
                pass  # keep layout aligned

        # Build tabs content
        with tabs:
            with gr.TabItem("Assistant Chat"):
                user_input = gr.Textbox(label="Ask HOLAS (text)", placeholder="Type your request...")
                send_btn = gr.Button("Send")
                response_out = gr.Textbox(label="HOLAS Response", lines=8, interactive=False)
                # Optional quick persona toggle (admin-only)
                persona_info = gr.Markdown("**Persona:** Open-minded; luxury tone enabled after admin unlock.")

            with gr.TabItem("Image Studio"):
                img_prompt = gr.Textbox(label="Image prompt", placeholder="Describe visuals...")
                img_steps = gr.Slider(5, 50, value=28, label="Steps")
                gen_img_btn = gr.Button("Generate Image")
                gallery = gr.Gallery(label="Generated Images").style(grid=2, height="300px")
                last_image_link = gr.Textbox(label="Public Image Link (if available)")

            with gr.TabItem("Website / Publish"):
                gr.Markdown("**Publish to WordPress (safe)**")
                post_title = gr.Textbox(label="Post title")
                post_content = gr.Textbox(label="Post content", lines=8)
                attach_image = gr.Textbox(label="Attach image local path (optional)", placeholder="e.g. media/holas_img_123.png")
                preview_btn = gr.Button("Dry-run Preview")
                preview_out = gr.Code(label="Preview")
                confirm_phrase = gr.Textbox(label="Confirm phrase (type: PUBLISH NOW)")
                publish_pin = gr.Textbox(label="PIN for publish", type="password")
                publish_btn = gr.Button("Publish Now")
                publish_result = gr.JSON(label="Publish Result")

            with gr.TabItem("Admin Panel"):
                gr.Markdown("**Admin Controls**")
                cfg = read_config()
                current_pin_box = gr.Textbox(label="Current PIN", value=cfg.get("pin"))
                new_pin_box = gr.Textbox(label="New PIN", type="password")
                save_pin_btn = gr.Button("Save PIN")
                # model switcher
                model_box = gr.Textbox(label="Model (HF name)", value=cfg.get("model"))
                switch_model_btn = gr.Button("Switch Model")
                # audit log viewer
                logs = gr.JSON(value=read_audit(), label="Audit Log (recent)")

        # ---------- callbacks ----------
        def try_unlock(pin):
            cfg = read_config()
            if pin == cfg.get("pin"):
                audit("admin", "unlock_success", {"method":"pin"})
                # prepare startup audio bytes (placeholder short chime)
                audio_bytes = get_builtin_chime_bytes()
                return "Unlocked — Welcome, Achiever.", True, (startup_html.update(visible=True), audio_bytes)
            else:
                audit("admin", "unlock_failed", {"attempt": pin})
                return "Invalid PIN.", gr.update(visible=False), None

        unlock_btn.click(try_unlock, inputs=[pin_in], outputs=[unlock_status, tabs, startup_audio])

        # Chat action
        def on_send(text):
            if not text or text.strip()=="":
                return "Type something — I'm ready."
            # safety checks
            low = text.lower()
            if any(t in low for t in ["hack","exploit","bypass","crack"]):
                audit("holas", "refused", {"prompt": text[:200]})
                return "I can't assist with hacking or bypassing systems. I can provide security best practices or connect you with vetted professionals."
            return generate_text(text)

        send_btn.click(on_send, inputs=[user_input], outputs=[response_out])

        # Image generation
        def on_generate_image(prompt, steps):
            if not prompt or prompt.strip()=="":
                return [], "No prompt provided."
            img, local_path, public_link_or_err = generate_image(prompt, steps=int(steps))
            if img is None:
                return [], f"Image generation failed: {public_link_or_err}"
            # Return gallery as list of images (PIL objects or file paths accepted)
            return [local_path], public_link_or_err or ""
        gen_img_btn.click(on_generate_image, inputs=[img_prompt, img_steps], outputs=[gallery, last_image_link])

        # Dry-run preview
        def on_preview(title, content, attach):
            preview = dry_run_publish(title, content, attach)
            return json.dumps(preview, indent=2)
        preview_btn.click(on_preview, inputs=[post_title, post_content, attach_image], outputs=[preview_out])

        # Publish action (requires confirm phrase and correct PIN)
        def on_publish(title, content, attach, confirm_phrase, pin):
            cfg = read_config()
            if confirm_phrase.strip() != "PUBLISH NOW":
                return {"success": False, "error": "Missing confirmation phrase 'PUBLISH NOW'."}
            if pin != cfg.get("pin"):
                return {"success": False, "error": "Invalid PIN."}
            # If WP creds present and ALLOW_REAL_PUBLISH true, attempt real publish
            res = wp_publish_post(title, content, attach)
            return res
        publish_btn.click(on_publish, inputs=[post_title, post_content, attach_image, confirm_phrase, publish_pin], outputs=[publish_result])

        # Admin: save new PIN
        def on_save_pin(new_pin):
            if not new_pin or len(new_pin.strip()) < 4:
                return {"status": "error", "message": "PIN must be at least 4 chars."}
            cfg = read_config()
            cfg["pin"] = new_pin.strip()
            write_config(cfg)
            audit("admin", "pin_changed", {"by":"admin"})
            return {"status": "ok", "message": "PIN updated."}
        save_pin_btn.click(on_save_pin, inputs=[new_pin_box], outputs=[logs])

        # Admin: switch model
        def on_switch_model(new_model):
            ok, msg = switch_model(new_model.strip())
            return {"ok": ok, "message": msg}
        switch_model_btn.click(on_switch_model, inputs=[model_box], outputs=[logs])

        # Refresh logs
        def on_refresh_logs():
            return read_audit()
        # attach refresh to logs viewer (user can reload Admin tab)
        # (we wire new actions to the same logs output)
        # done via save_pin_btn and switch_model_btn returning logs

    return demo

# ---------- small helper functions for animation & chime ----------
def STARTUP_ANIMATION_HTML():
    # Inline CSS + minimal animation - safe to embed in Gradio
    html = """
    <div style="width:100%;padding:18px;border-radius:12px;background:linear-gradient(90deg,#07152f,#021);text-align:center;color:#FFD700;">
      <div style="font-size:36px;font-weight:700">HOLAS ⚜️</div>
      <div style="color:#0B2D8D;margin-top:6px">Initializing Luxury Intelligence — please wait</div>
    </div>
    """
    return html

# small base64 chime for startup (placeholder short sound)
def get_builtin_chime_bytes():
    # Very small base64 placeholder (may be non-musical); replace with proper audio if desired
    b64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjI0LjEwNQAAAAAAAAAAAAAA"
    try:
        return b64decode(b64)
    except Exception:
        return None

# ---------- Entrypoint ----------
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
