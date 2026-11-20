# Team AI Engineering Assistant — Fred (`discord-bot/`)

This directory contains **Fred**, the autonomous AI engineering teammate built for the CSC 434 Final Project team (Muamer and Michael/Mike).

---

## 1. Overview & Architecture

- **Role**: Active 3rd team member providing real-time AI/ML architectural feedback, hyperparameter guidance, code review, and project coordination.
- **Engine**: Powered by Groq's high-speed cloud inference (`AsyncOpenAI` client) using high-parameter open-weights LLMs.
- **Persistent Brain (`PROJECT_MEMORY.md`)**: Whenever architectural decisions, model topology updates, dataset changes, or task assignments are discussed, Fred automatically updates `PROJECT_MEMORY.md` with structured bullet points.

---

## 2. Directory Contents

```text
discord-bot/
├── bot.py             # Discord client with LLM integration and automatic memory persistence
├── PROJECT_MEMORY.md  # Shared persistent engineering memory, milestones, and decisions log
└── README.md          # Documentation and setup instructions
```

---

## 3. Setup & Execution

### Prerequisites
Create a `.env` file in `discord-bot/` (or repository root):
```env
DISCORD_TOKEN=your_discord_bot_token_here
LLM_API_KEY=your_groq_or_openai_api_key_here
```

### Running the Bot
```bash
# Activate environment
source ../venv/bin/activate  # or venv\Scripts\activate on Windows

# Install discord dependencies if needed
pip install discord.py python-dotenv openai

# Start Fred
python bot.py
```

---

## 4. Key Capabilities

1. **Automatic Memory Updates**: Extracts `<memory_update>` blocks from conversations and persists them directly into `PROJECT_MEMORY.md`.
2. **Context-Aware Dialogue**: Feeds recent message history (up to 20 turns) along with the current `PROJECT_MEMORY.md` state into the prompt.
3. **Multi-Modal Attachment Reading**: Automatically inspects and parses code attachments (`.py`, `.json`, `.csv`, `.md`, `.yaml`) sent by teammates in Discord channels.
4. **NBA AI Referee Synchronization**: Tracks all phases of the v3.0 full-stack pipeline, including E-BARD basketball perception (`player`/`referee`/`basketball`/`hoop`), 2-team CIELAB jersey clustering, 16-D depth-gated Rule 12 kinematics, Video Swin3D-T + Gated Attention MIL training, leak-free game-grouped CV, dedicated test isolation (`data/test/`), and autonomous drop-folder video annotation (`--annotate-test`).
