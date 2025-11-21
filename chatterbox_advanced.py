
import os
import sys
import time
import shutil
import subprocess
import yaml
import modal

from fastapi import FastAPI, Request
from fastapi.responses import (
    StreamingResponse,
    HTMLResponse,
    JSONResponse,
    FileResponse,
)
from starlette.background import BackgroundTask

# =====================================================================================
# 1. Persistent Storage (Modal Volume)
# =====================================================================================

vol = modal.Volume.from_name("chatterbox-data", create_if_missing=True)

# =====================================================================================
# 2. Build Image (clone from GitHub)
# =====================================================================================

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "curl")
    # outer proxy deps
    .pip_install(
        "uvicorn",
        "fastapi",
        "requests",
        "jinja2",
        "python-multipart",
        "soundfile",
        "httpx",
        "pyyaml",
    )
    # Clone from GitHub master branch (has async files)
    .run_commands("rm -rf /app && git clone --branch master https://github.com/DaniyalAfzaal/ChatterboxDANNYX /app")
    .run_commands("cd /app && pip install -r requirements-nvidia.txt")
    .run_commands("chmod +x /app/server.py")
)

app = modal.App("chatterbox-final")

# =====================================================================================
# 3. Helpers: Persistence, Config, UI Patch, config.py & models.py patches
# =====================================================================================

def setup_persistence():
    """
    Mounts the persistent volume and creates symlinks for the app's internal paths.
    /app/outputs -> /data/outputs
    /app/model_cache -> /data/model_cache
    /app/voices -> /data/voices
    /app/reference_audio -> /data/reference_audio
    """
    folders = ["outputs", "model_cache", "voices", "reference_audio", "presets"]
    for folder in folders:
        os.makedirs(f"/data/{folder}", exist_ok=True)
        app_path = f"/app/{folder}"
        # Remove real directory (not symlink) so we can safely symlink
        if os.path.exists(app_path) and not os.path.islink(app_path):
            shutil.rmtree(app_path)
        if not os.path.exists(app_path):
            os.symlink(f"/data/{folder}", app_path)


def enforce_config():
    """
    Minimal config.yaml override to ensure:
      - Server listens on 0.0.0.0:8000
      - Uses /data paths for outputs & cache
    """
    config_path = "/app/config.yaml"
    base_conf = {
        "server": {"host": "0.0.0.0", "port": 8000},
        "tts_device": "cuda",
        "model_cache_path": "/data/model_cache",
        "output_path": "/data/outputs",
        "generation_defaults": {
            "speed": 1.0,
            "temperature": 0.7,
            "split_text": True,
            "chunk_size": 1800,
        },
        "ui_state": {},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(base_conf, f)
    print("✅ Wrote minimal config.yaml for port 8000.")


def patch_main_ui_js():
    """
    FIX: Prevent 'isUserInteraction is not defined' error by injecting
    a global variable into index.html before script.js runs.
    """
    index_path = "/app/ui/index.html"
    if not os.path.exists(index_path):
        print("⚠️ ui/index.html not found; skipping UI patch.")
        return

    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "window.isUserInteraction" not in content:
        # Inject at the very top of head to ensure it's available early
        if "<head>" in content:
            content = content.replace(
                "<head>",
                '<head>\n    <script>window.isUserInteraction = true;</script>',
                1
            )
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("✅ Patched index.html with JS fix.")
        else:
             print("⚠️ Could not find <head> tag in index.html to patch.")
    else:
        print("ℹ️ index.html already patched for isUserInteraction.")


def patch_config_py():
    """
    Ensure /app/config.py defines all helpers expected by server.py.
    Also patches YamlConfigManager to add missing get_dict method.
    """
    cfg_path = "/app/config.py"
    if not os.path.exists(cfg_path):
        print("⚠️ config.py not found; skipping config patch.")
        return

    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()

    modified = False

    # --- Part 1: Add get_dict to YamlConfigManager if missing ---
    if "def get_dict(self, key_path: str" not in content:
        # We'll look for the end of the get_bool method to insert get_dict after it
        search_str = """            return False

    def get_path("""
    
        replace_str = """            return False

    def get_dict(self, key_path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raw_value = self.get(key_path)
        if raw_value is None:
            return default if default is not None else {}
        if isinstance(raw_value, dict):
            return raw_value
        return default if isinstance(default, dict) else {}

    def update(self, new_settings: Dict[str, Any]):
        with self._lock:
            _deep_merge_dicts(new_settings, self.config)
            self.config = self._resolve_paths_and_device(self.config)

    def save(self) -> bool:
        return self.save_config_yaml()

    def get_path("""
        
        if search_str in content:
            content = content.replace(search_str, replace_str)
            print("✅ Patched config.py: Added get_dict to YamlConfigManager.")
            modified = True
        else:
            print("⚠️ Could not find insertion point for get_dict in YamlConfigManager.")
    else:
        print("ℹ️ config.py already has get_dict.")

    # --- Part 2: Add helper stubs at the end of file ---
    if "def get_base_output_dir" in content:
        print("ℹ️ config.py already has get_base_output_dir; skipping stubs patch.")
    else:
        patch_block = r'''

# ---- Modal patch: add config helper stubs ----
import os as _os
import yaml as _yaml
from typing import Any as _Any, Mapping as _Mapping

_CONFIG_PATH = _os.environ.get("CHATTERBOX_CONFIG_PATH", "/app/config.yaml")

def get_config_path() -> str:
    return _CONFIG_PATH

def _load_yaml_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}
    if not isinstance(cfg, dict):
        return {}
    return cfg

def get_config() -> dict:
    return _load_yaml_config()

def _unwrap_cfg(cfg: _Any) -> dict:
    if cfg is None:
        cfg = get_config()
    if isinstance(cfg, _Mapping):
        return dict(cfg)
    maybe_cfg = getattr(cfg, "config", None)
    if isinstance(maybe_cfg, _Mapping):
        return dict(maybe_cfg)
    return {}

def get_base_output_dir(cfg: _Any = None) -> str:
    cfg_map = _unwrap_cfg(cfg)
    out = cfg_map.get("output_path")
    if not out:
        server = cfg_map.get("server") or {}
        out = server.get("output_path") or "/app/outputs"
    _os.makedirs(out, exist_ok=True)
    return out

def get_server_host(cfg: _Any = None) -> str:
    cfg_map = _unwrap_cfg(cfg)
    server = cfg_map.get("server") or {}
    return server.get("host", "0.0.0.0")

def get_server_port(cfg: _Any = None) -> int:
    cfg_map = _unwrap_cfg(cfg)
    server = cfg_map.get("server") or {}
    try:
        return int(server.get("port", 8000))
    except Exception:
        return 8000

def get_device_setting(cfg: _Any = None) -> str:
    cfg_map = _unwrap_cfg(cfg)
    return cfg_map.get("tts_device", "cuda")

def get_default_split_text_setting(cfg: _Any = None) -> bool:
    cfg_map = _unwrap_cfg(cfg)
    gen = cfg_map.get("generation_defaults") or {}
    if isinstance(gen, _Mapping) and "split_text" in gen:
        return bool(gen["split_text"])
    if "split_text" in cfg_map:
        return bool(cfg_map["split_text"])
    return True

def get_default_chunk_size_setting(cfg: _Any = None) -> int:
    cfg_map = _unwrap_cfg(cfg)
    gen = cfg_map.get("generation_defaults") or {}
    if isinstance(gen, _Mapping) and "chunk_size" in gen:
        try:
            return int(gen["chunk_size"])
        except Exception:
            pass
    if "chunk_size" in cfg_map:
        try:
            return int(cfg_map["chunk_size"])
        except Exception:
            pass
    return 1800

def get_default_split_text(cfg: _Any = None) -> bool:
    return get_default_split_text_setting(cfg)

def get_default_chunk_size(cfg: _Any = None) -> int:
    return get_default_chunk_size_setting(cfg)

'''
        content += patch_block
        print("✅ Patched config.py with helper stubs.")
        modified = True
    
    # Write back changes if any
    if modified:
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(content)


def patch_server_py():
    """
    FIX: server.py serves styles.css and script.js as PlainTextResponse (text/plain).
    Browsers ignore CSS with text/plain MIME type.
    We patch it to use FileResponse with correct media_type.
    """
    server_path = "/app/server.py"
    if not os.path.exists(server_path):
        print("⚠️ server.py not found; skipping server patch.")
        return

    with open(server_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Patch styles.css route
    old_styles_route = """@app.get("/styles.css", response_class=PlainTextResponse)
async def styles():
    css_path = Path("ui") / "styles.css"
    if not css_path.is_file():
        return PlainTextResponse(content="", status_code=404)
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    return PlainTextResponse(content=css_content, status_code=200)"""

    new_styles_route = """@app.get("/styles.css")
async def styles():
    css_path = Path("ui") / "styles.css"
    if not css_path.is_file():
        return PlainTextResponse(content="", status_code=404)
    return FileResponse(css_path, media_type="text/css")"""

    if old_styles_route in content:
        content = content.replace(old_styles_route, new_styles_route)
        print("✅ Patched server.py: Fixed styles.css Content-Type.")
    else:
        print("⚠️ Could not find exact styles.css route in server.py to patch.")

    # Patch script.js route
    old_js_route = """@app.get("/script.js", response_class=PlainTextResponse)
async def script():
    js_path = Path("ui") / "script.js"
    if not js_path.is_file():
        return PlainTextResponse(content="// script.js not found", status_code=404)
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()
    return PlainTextResponse(content=js_content, status_code=200)"""

    new_js_route = """@app.get("/script.js")
async def script():
    js_path = Path("ui") / "script.js"
    if not js_path.is_file():
        return PlainTextResponse(content="// script.js not found", status_code=404)
    return FileResponse(js_path, media_type="application/javascript")"""

    if old_js_route in content:
        content = content.replace(old_js_route, new_js_route)
        print("✅ Patched server.py: Fixed script.js Content-Type.")
    else:
        print("⚠️ Could not find exact script.js route in server.py to patch.")

    # Patch list_predefined_voices to return voice objects instead of strings
    old_list_voices_func = """def list_predefined_voices() -> List[str]:
    voices_dir = get_predefined_voices_path(ensure_absolute=True)
    if not voices_dir.exists():
        return []
    return utils.get_valid_reference_files(voices_dir)"""

    new_list_voices_func = """def list_predefined_voices() -> List[Dict[str, str]]:
    voices_dir = get_predefined_voices_path(ensure_absolute=True)
    if not voices_dir.exists():
        return []
    return utils.get_predefined_voices()"""

    if old_list_voices_func in content:
        content = content.replace(old_list_voices_func, new_list_voices_func)
        print("✅ Patched server.py: Fixed list_predefined_voices to return voice objects.")
    else:
        print("⚠️ Could not find exact list_predefined_voices function to patch.")

    # Patch the /get_predefined_voices endpoint to return voice objects directly
    old_get_voices_endpoint = """@app.get("/get_predefined_voices", response_model=FileListResponse)
async def get_predefined_voices():
    try:
        files = list_predefined_voices()
        return FileListResponse(files=files)"""

    new_get_voices_endpoint = """@app.get("/get_predefined_voices")
async def get_predefined_voices():
    try:
        voices = list_predefined_voices()
        return voices"""

    if old_get_voices_endpoint in content:
        content = content.replace(old_get_voices_endpoint, new_get_voices_endpoint)
        print("✅ Patched server.py: Fixed /get_predefined_voices endpoint.")
    else:
        print("⚠️ Could not find exact /get_predefined_voices endpoint to patch.")

    with open(server_path, "w", encoding="utf-8") as f:
        f.write(content)
def patch_models_py():
    """
    Ensure /app/models.py defines UIInitialDataResponse AND FileListResponse.
    Also fix UpdateStatusResponse to match server.py usage.
    """
    models_path = "/app/models.py"
    if not os.path.exists(models_path):
        print("⚠️ models.py not found; creating it.")
        with open(models_path, "w", encoding="utf-8") as f:
            f.write("from pydantic import BaseModel\nfrom typing import List, Dict, Any, Optional\n\n")

    with open(models_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "from typing import" not in content:
        content = "from typing import List, Dict, Any, Optional\n" + content
    if "from pydantic import" not in content:
        content = "from pydantic import BaseModel\n" + content

    update_status_code = """
class UpdateStatusResponse(BaseModel):
    success: bool
    message: str
    restart_required: bool
"""
    
    ui_initial_data_code = """
class UIInitialDataResponse(BaseModel):
    reference_files: List[str]
    predefined_voices: List[Dict[str, Any]]
    ui_state: Dict[str, Any]
    presets: List[Dict[str, Any]]
"""

    file_list_code = """
class FileListResponse(BaseModel):
    files: List[str] = []
"""

    content += "\n" + update_status_code + "\n" + ui_initial_data_code + "\n" + file_list_code

    with open(models_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Patched models.py with correct response models.")


def patch_utils_py():
    """
    FIX: TypeError: get_valid_reference_files() takes 0 positional arguments but 1 was given.
    This patches utils.py to accept *args, **kwargs so it doesn't crash when server.py passes an argument.
    """
    utils_path = "/app/utils.py"
    if not os.path.exists(utils_path):
        print("⚠️ utils.py not found; skipping utils patch.")
        return

    with open(utils_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if already fixed (either by git pull or previous patch)
    if "def get_valid_reference_files(ref_audio_dir_path" in content or "def get_valid_reference_files(*args" in content:
        print("ℹ️ utils.py already has fixed signature.")
        return

    # Replace the signature to accept arguments and ignore them (safe fallback)
    old_sig = "def get_valid_reference_files() -> List[str]:"
    new_sig = "def get_valid_reference_files(*args, **kwargs) -> List[str]:"
    
    if old_sig in content:
        content = content.replace(old_sig, new_sig)
        with open(utils_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Patched utils.py to fix get_valid_reference_files signature.")
    else:
        print("⚠️ Could not find exact signature in utils.py to patch. Assuming it might be different.")


# =====================================================================================
# 4. Simple HTML Dashboard for Downloading Generated Files
# =====================================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatterbox File Manager</title>
    <style>
        body { font-family: sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; max-width: 800px; margin: 0 auto; }
        .card { background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #334155; }
        h1 { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 10px;}
        button { background: #10b981; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .file-list { list-style: none; padding: 0; }
        .file-item { display: flex; justify-content: space-between; padding: 12px; border-bottom: 1px solid #334155; align-items: center; }
        .file-item a { color: white; text-decoration: none; background: #3b82f6; padding: 8px 15px; border-radius: 4px; font-size: 0.9em;}
    </style>
</head>
<body>
    <h1>
        <span>📂 File Manager</span>
        <a href="/" class="main-link" style="color:#3b82f6; border: 1px solid; padding:6px 10px; border-radius:4px;">← Go to Generator</a>
    </h1>
    
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 15px;">
            <span style="color:#94a3b8">Files saved from the Generator UI or /tts_job:</span>
            <button onclick="loadFiles()">↻ Refresh List</button>
        </div>
        <ul id="file_list"></ul>
    </div>

    <script>
        async function loadFiles() {
            const list = document.getElementById('file_list');
            list.innerHTML = '<div style="padding:20px; text-align:center; color:#64748b">Scanning storage...</div>';
            
            try {
                const res = await fetch('/custom-api/files');
                const files = await res.json();
                list.innerHTML = '';
                
                if (!files || files.length === 0) {
                    list.innerHTML = '<li style="padding:10px">No files found. Generate audio on the main page or via /tts_job.</li>';
                    return;
                }
                
                files.forEach(f => {
                    const li = document.createElement('li');
                    li.className = 'file-item';
                    const displayName = f.split('/').pop(); 
                    li.innerHTML = `<span>${displayName}</span> <a href="/custom-api/download?file=${encodeURIComponent(f)}" target="_blank">Download</a>`;
                    list.appendChild(li);
                });
            } catch(e) { 
                console.error(e);
                list.innerHTML = '<div style="color:red; padding:10px">Error loading files. Is the server running?</div>';
            }
        }
        loadFiles();
    </script>
</body>
</html>
"""

# =====================================================================================
# 5. Main Modal Function (ASGI app)
# =====================================================================================

@app.function(
    image=image,
    gpu="L4",                # L4 GPU for TTS
    volumes={"/data": vol},  # Persistent volume
    timeout=3600,            # 1 hour per container run
    max_containers=1,        # Single container
)
@modal.asgi_app()
def entrypoint():
    # --- Bootstrap: persistence + repo setup ---
    setup_persistence()
    sys.path.append("/app")
    os.chdir("/app")
    enforce_config()
    patch_main_ui_js()
    patch_config_py()
    patch_server_py()  # <--- Added this new patch
    patch_models_py()
    patch_utils_py()  # <--- Added this new patch

    # --- Start the original Chatterbox server on 127.0.0.1:8000 ---
    print("🚀 Starting internal TTS server...")
    subprocess.Popen(
        ["python", "server.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # --- Wait until internal FastAPI (/docs) is up ---
    import requests as _requests

    for i in range(60):
        try:
            r = _requests.get("http://127.0.0.1:8000/docs", timeout=1)
            if r.status_code == 200:
                print("✅ Internal TTS server online at 127.0.0.1:8000")
                break
        except Exception:
            time.sleep(1)
    else:
        print("⚠️ Internal server did not respond in time; proxy will still start and show 502 until ready.")

    # --- Build outer FastAPI app that Modal will expose ---
    web_app = FastAPI()
    import httpx

    client = httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=600.0)

    @web_app.get("/manager", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD_HTML

    @web_app.get("/custom-api/files")
    def list_files():
        """
        Scan /data/outputs recursively for .wav/.mp3/.opus files.
        Returns relative paths sorted by modification time (newest first).
        """
        vol.reload()
        root_dir = "/data/outputs"
        if not os.path.exists(root_dir):
            return []

        found_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith((".wav", ".mp3", ".opus")):
                    full_path = os.path.join(dirpath, f)
                    rel_path = os.path.relpath(full_path, root_dir)
                    found_files.append(rel_path)

        found_files.sort(
            key=lambda rel: os.path.getmtime(os.path.join(root_dir, rel)),
            reverse=True,
        )
        return found_files

    @web_app.get("/custom-api/download")
    def download_file(file: str):
        """
        Download a previously saved output file from /data/outputs.
        """
        vol.reload()
        clean_path = os.path.normpath(file).lstrip("/")
        file_path = os.path.join("/data/outputs", clean_path)

        if not file_path.startswith("/data/outputs/"):
            return JSONResponse({"error": "Invalid file path"}, status_code=400)

        if os.path.exists(file_path):
            return FileResponse(
                file_path,
                filename=os.path.basename(file_path),
            )
        return JSONResponse({"error": "File not found"}, status_code=404)

    @web_app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def proxy(request: Request, path_name: str):
        """
        Generic reverse proxy to the internal Chatterbox server:
          - UI:       /, /script.js, /styles.css, /ui/*
          - API:      /tts, /tts_job, /v1/audio/speech, etc.
        """
        if path_name.startswith("custom-api") or path_name == "manager":
            return JSONResponse({"error": "Not Found"}, status_code=404)

        url = httpx.URL(
            path="/" + path_name if not path_name.startswith("/") else path_name,
            query=request.url.query.encode("utf-8"),
        )

        body = await request.body()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ["host", "content-length"]
        }

        try:
            proxied_req = client.build_request(
                request.method,
                url,
                headers=headers,
                content=body,
            )
            resp = await client.send(proxied_req, stream=True)

            return StreamingResponse(
                resp.aiter_raw(),
                status_code=resp.status_code,
                headers={k: v for k, v in resp.headers.items() if k.lower() != "content-length"},
                background=BackgroundTask(resp.aclose),
            )
        except Exception as e:
            print("❌ Proxy error:", e)
            return JSONResponse({"error": "Server Loading or Internal Error"}, status_code=502)

    return web_app
