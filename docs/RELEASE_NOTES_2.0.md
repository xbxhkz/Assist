# Assist 2.0

**Assist is an AI assistant that runs on your own Windows PC.** You chat with it in
plain language, and it can also *do* things for you — search the web, work with your
files, generate images, read live equipment data, manage email and calendar, and more.
Because it can run entirely offline, your conversations and your AI models can stay on
your machine. No account is required to get started.

If you've never used a tool like this, don't worry — the **"New to Assist — or to AI?
Start here"** section inside the app's **Help** menu walks you through your very first
chat, and this page explains everything in plain language too.

---

## What is this, in one minute?

- **You talk, it answers.** Type a question at the bottom of the screen and press Enter.
- **The "model" is the brain.** Before you can chat you need one AI *model*. You can
  **download one for free and run it offline** (recommended to start), or **connect an
  online provider** (OpenAI, Anthropic, etc.) with an account.
- **Chat vs. Agent.** *Chat* just talks. *Agent* lets the AI take actions for you, asking
  permission before anything sensitive. There's a switch next to the chat box.
- **You're in control.** The most powerful abilities (running commands, controlling your
  mouse/keyboard, reading your screen) are **off by default** and only for an admin
  account. You turn them on deliberately.

## Install & your first chat

1. Download **`Assist-Setup.exe`** from this release and run it. It's a normal Windows
   installer — no Docker, no command line, no cloud account.
2. Launch **Assist**. On first run it will ask you to create a local login.
3. Click **Local Models** in the left sidebar. Pick a suggested model (the list is sorted
   so models that fit *your* computer are near the top) and click **Download**. Models are
   large, so give it a few minutes.
4. Click **Serve** — choose **GPU** if your PC has a graphics card, otherwise **CPU** —
   and wait until it says it's running.
5. Type a question in the box at the bottom and press **Enter**. You're chatting. 🎉

Prefer an online model instead? Type `/setup` in the chat box to connect a provider.

---

## New in 2.0

**🖼️ Just ask for a picture — Assist starts the image model for you.**
Previously you had to manually start an image model before generating. Now: set a default
image model under **Settings → Image generation**, then simply ask for an image while
chatting with any normal model ("draw a red bicycle in the rain"), and Assist
automatically starts the image model and creates it. Finished images land in the Gallery.
Supports local, offline diffusion models (FLUX.1 / FLUX.2 klein / SDXL / Chroma / Z-Image)
with automatic GPU→CPU fallback.

**🏭 Industrial Assistant — for maintenance techs and engineers.** *(admin-only)*
- **Diagnose from a photo.** Attach a fault-code screen, wiring/ladder schematic, VFD/drive
  display, nameplate, or thermal image and ask what's wrong — you get maintenance-expert,
  *safety-first* guidance.
- **Read live equipment data.** On your private network, read live values from a PLC,
  meter, or drive over **Modbus TCP** or **OPC UA** (strictly read-only; private addresses
  only, for safety).
- **Cite your manuals.** Add your equipment manuals/datasheets to a searchable library and
  the assistant quotes the exact **manual and page** while diagnosing.

**🧩 Workflows — build automations as a flowchart.** *(admin)*
Drag out steps (ask the AI, run a tool, branch on a condition), connect them, and run the
whole thing on demand, on a schedule, on an event, or from a webhook. Great for repeatable
jobs like "every morning, summarize new emails into a note."

**⚙️ Smarter local model serving.** The amount of context a served model uses is now
matched automatically to the model and to your graphics memory (instead of a fixed value),
so small graphics cards keep more of the model on the GPU and stay faster.

**🎨 Refreshed app icon and logo.**

## Everything Assist can do (feature tour)

- **Chat & Agent** with any local or online model; per-turn tool toggles; slash commands
  (type `/`).
- **Local model serving** — search Hugging Face, download GGUF models, serve on CPU/GPU
  with automatic VRAM fitting; external EXL2 (TabbyAPI) and MLX endpoints too.
- **Image generation & Gallery** — local diffusion, LoRA support, albums, and an image
  editor (crop, adjust, background removal).
- **Voice conversations** — hands-free talk using a local Whisper speech model; it listens,
  transcribes, replies aloud, and re-opens the mic for you.
- **Desktop control & the AI Operator** *(admin)* — launch apps, find files, manage
  windows, read the screen, drive the mouse/keyboard, and run goal-driven tasks with a
  confirmation before each action.
- **Shell commands** *(admin)* — run PowerShell/cmd; read-only runs automatically,
  anything that changes state asks for your approval first.
- **Email, Calendar & Tasks** — connect an account to read/search/send mail; schedule
  background tasks ("every weekday at 9am check for urgent emails").
- **Notes, Library & Brain** — quick notes, a document library with an editor and PDF
  search, and persistent memory the assistant builds about you.
- **Coding** — grant folder access (per-chat Workspace or standing folders), then have the
  agent read/write/edit code and open results in VS Code.
- **Connectors** — link Gmail, Google Calendar, Google Drive, Hugging Face, and other
  services via the Model Context Protocol (MCP).
- **Hardware monitor** — live CPU / RAM / VRAM graphs.

## Good to know

- **Windows 10/11**, 64-bit. A graphics card (GPU) makes local models much faster but
  isn't required — you can run on CPU.
- **Your data stays local** by default; it only leaves your machine if you connect an
  online provider or a cloud connector.
- **Logs** for troubleshooting live in the data folder (`~/.assist/data/logs`).
- **Getting help:** open the **Help** menu in the app (Manual, Tutorials, About), or type
  `/demo` in the chat box for a full guided tour.

---

*Assist 2.0 — a local-first AI workspace. Your data and your models stay on your machine.*
