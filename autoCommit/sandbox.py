from pathlib import Path
import os
import pty
import subprocess

prompt = "日本の首都は？"

cmd = [
    str(Path("~/.gac/bin/llama-cli").expanduser()),
    "-m",
    str(Path("~/.gac/models/gemma-3-1b-it-Q4_K_M.gguf").expanduser()),
    "-no-cnv",
    "-st",
    "-p",
    prompt,
]

master_fd, slave_fd = pty.openpty()

process = subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,
    stdout=slave_fd,
    stderr=slave_fd,
    text=True,
)

os.close(slave_fd)

output = b""

while True:
    try:
        data = os.read(master_fd, 1024)
        if not data:
            break
        output += data
    except OSError:
        break

process.wait()
os.close(master_fd)

text = output.decode(errors="ignore")

print("text", text)