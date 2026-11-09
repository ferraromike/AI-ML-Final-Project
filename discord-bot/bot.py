import os
import re
from datetime import datetime
from pathlib import Path
import discord
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Always resolve paths relative to bot.py
load_dotenv()
MEMORY_FILE = Path("PROJECT_MEMORY.md")

# --- Groq Free Cloud Configuration ---
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "openai/gpt-oss-120b"

ai_client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=BASE_URL,
)

HISTORY_LIMIT = 20
ALLOWED_EXTENSIONS = (".py", ".txt", ".md", ".json", ".csv", ".yaml")

FRED_INSTRUCTIONS = """You are Fred, the 3rd member of a two-person student engineering team in CSC 434: Artificial Intelligence.
Messages from your human teammates are formatted as '[Name]: message'.
Address teammates by name when relevant, act like a sharp, collaborative peer, and keep answers technical, concrete, and concise.

### AUTOMATIC MEMORY UPDATE RULE (CRITICAL):
You have a persistent brain stored in `PROJECT_MEMORY.md` (shown below).
Whenever the user's prompt or your response introduces, modifies, or finalizes ANY information that pertains to the project—such as:
- Architectural decisions, state/action changes, or reward functions
- Hyperparameters, model topologies, or algorithm choices
- Code structure, class/function names, or bugs fixed
- Task division, deadlines, or report/presentation plans

You MUST append a `<memory_update>` block at the VERY END of your response containing 1 to 3 concise markdown bullet points summarizing the new project facts to save.
Format example:
<memory_update>
- Decided to use LeakyReLU activations in the hidden layer of the GA network.
- Assigned Flappy Bird raycast wrapper implementation to Mike.
</memory_update>

If the prompt is purely casual banter or repeats facts already in `PROJECT_MEMORY.md` with nothing new to log, omit the `<memory_update>` tag."""

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)


def load_project_memory() -> str:
    """Reads PROJECT_MEMORY.md from disk."""
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(
            "# CSC 434 AI/ML Final Project Memory — Fred\n\n## 5. Active Decisions, Hyperparameters & Tasks Log\n",
            encoding="utf-8",
        )
    return MEMORY_FILE.read_text(encoding="utf-8")


def append_to_project_memory(author_name: str, update_text: str) -> None:
    """Appends new bullet points extracted by Fred to PROJECT_MEMORY.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    cleaned_lines = []
    for line in update_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("-"):
            line = f"- {line}"
        cleaned_lines.append(f"{line} *(logged via {author_name}, {timestamp})*")

    if cleaned_lines:
        entry = "\n" + "\n".join(cleaned_lines) + "\n"
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(entry)


async def extract_message_text(msg: discord.Message, bot_id: int) -> str:
    """Cleans @mentions and reads any attached code/text files."""
    text = msg.content.replace(f"<@{bot_id}>", "@Fred").replace(f"<@!{bot_id}>", "@Fred").strip()
    for attachment in msg.attachments:
        if attachment.filename.endswith(ALLOWED_EXTENSIONS) and attachment.size < 80_000:
            try:
                file_bytes = await attachment.read()
                file_content = file_bytes.decode("utf-8", errors="ignore")
                text += f"\n\n--- Attached File: {attachment.filename} ---\n{file_content}\n--- End File ---"
            except Exception:
                pass
    return text


@bot.event
async def on_ready():
    print(f"✅ Fred is online as {bot.user} | Model: {MODEL_NAME}")
    print(f"🧠 Watching & updating memory file: {MEMORY_FILE}")


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    content = message.content.strip()

    # --- COMMAND: !memory (Inspect Fred's current .md file) ---
    if content == "!memory":
        mem = load_project_memory()
        chunk_size = 1800
        chunks = [mem[i : i + chunk_size] for i in range(0, len(mem), chunk_size)]
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks, 1):
            msg_text = f"🧠 **`PROJECT_MEMORY.md` (Part {idx} of {total_chunks}):**\n```markdown\n{chunk}\n```"
            await message.reply(msg_text, mention_author=False)
        return

    # --- COMMAND: !compact (Deduplicate and clean up Section 5 of PROJECT_MEMORY.md) ---
    if content == "!compact":
        async with message.channel.typing():
            current_mem = load_project_memory()
            compact_prompt = (
                "You are maintaining our team's `PROJECT_MEMORY.md` file. "
                "Rewrite the markdown file below to merge duplicate notes in Section 5, "
                "organize recent decisions cleanly, and preserve all core specifications in Sections 1-4. "
                "Output ONLY the raw updated Markdown content (no backticks wrapping the whole file).\n\n"
                + current_mem
            )
            try:
                res = await ai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": compact_prompt}],
                    temperature=0.2,
                )
                updated_md = res.choices[0].message.content.strip()
                if updated_md and "# CSC 434" in updated_md:
                    MEMORY_FILE.write_text(updated_md + "\n", encoding="utf-8")
                    await message.reply("✨ **Fred compacted and organized `PROJECT_MEMORY.md`!**", mention_author=False)
                else:
                    await message.reply("⚠️ Compaction skipped to prevent overwriting headers.", mention_author=False)
            except Exception as e:
                await message.reply(f"⚠️ **Error compacting memory:** `{e}`", mention_author=False)
        return

    # --- STANDARD CHAT WITH FRED (@Fred or Reply to Fred) ---
    is_mentioned = bot.user in message.mentions
    is_reply_to_bot = (
        message.reference is not None
        and isinstance(message.reference.resolved, discord.Message)
        and message.reference.resolved.author == bot.user
    )

    if not (is_mentioned or is_reply_to_bot):
        return

    async with message.channel.typing():
        current_memory = load_project_memory()
        system_prompt = (
            f"{FRED_INSTRUCTIONS}\n\n"
            f"=== CURRENT PROJECT_MEMORY.md ===\n{current_memory}\n=== END PROJECT_MEMORY.md ==="
        )

        raw_history = [msg async for msg in message.channel.history(limit=HISTORY_LIMIT)]
        raw_history.reverse()

        formatted_messages = [{"role": "system", "content": system_prompt}]

        for msg in raw_history:
            clean_text = await extract_message_text(msg, bot.user.id)
            if not clean_text:
                continue
            if msg.author == bot.user:
                # Strip old footer badges from Fred's past messages to keep context clean
                clean_text = clean_text.replace("🧠 *Fred updated `PROJECT_MEMORY.md`*", "").strip()
                formatted_messages.append({"role": "assistant", "content": clean_text})
            else:
                formatted_messages.append({
                    "role": "user",
                    "content": f"[{msg.author.display_name}]: {clean_text}",
                })

        try:
            response = await ai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=formatted_messages,
                temperature=0.3,
            )
            raw_reply = response.choices[0].message.content or "*(No response)*"

            # Extract any <memory_update>...</memory_update> blocks
            memory_matches = re.findall(
                r"<memory_update>(.*?)</memory_update>", raw_reply, flags=re.DOTALL | re.IGNORECASE
            )
            clean_reply = re.sub(
                r"<memory_update>.*?</memory_update>", "", raw_reply, flags=re.DOTALL | re.IGNORECASE
            ).strip()

            if memory_matches:
                combined_updates = "\n".join(m.strip() for m in memory_matches if m.strip())
                if combined_updates:
                    append_to_project_memory(message.author.display_name, combined_updates)
                    clean_reply += "\n\n🧠 *Fred updated `PROJECT_MEMORY.md`*"

            for i in range(0, len(clean_reply), 1950):
                await message.reply(clean_reply[i : i + 1950], mention_author=False)

        except Exception as e:
            await message.reply(f"⚠️ **API Error:** `{e}`")


token = os.getenv("DISCORD_BOT_TOKEN")
if not token:
    raise ValueError("❌ DISCORD_BOT_TOKEN not found! Check your .env file.")
bot.run(token)
