import streamlit as st
import requests
import os
import uuid
import subprocess
import time
import gc
from groq import Groq

MODEL_ID = "whisper-large-v3-turbo"

st.set_page_config(page_title="播客转文字", page_icon="🎧")
st.title("🎧 播客转文字 (Groq 稳定版)")
st.info("💡 手机/电脑都适用：后台串行处理，每段完成后立即刷新，降低断线风险。")

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    api_key = st.text_input("请输入 Groq API Key (gsk_...)", type="password")
    if not api_key:
        st.stop()

def get_real_audio_url(url: str) -> str | None:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10, stream=True)
        content_type = resp.headers.get("Content-Type", "")
        if "audio" in content_type or url.endswith((".m4a", ".mp3")):
            return url
        import re
        match = re.search(r'(https?://[^\s"\'<>]+\.(?:m4a|mp3))', resp.text)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None

def transcribe_with_retry(client: Groq, chunk_file: str) -> str:
    for _ in range(3):
        try:
            with open(chunk_file, "rb") as f:
                text = client.audio.transcriptions.create(
                    file=(chunk_file, f.read()),
                    model=MODEL_ID,
                    language="zh",
                    response_format="text"
                )
            return text.encode("utf-8", "ignore").decode("utf-8")
        except Exception:
            time.sleep(2)
    return "[该片段转写失败]"

def process_audio(input_url: str):
    client = Groq(api_key=api_key)

    status_box = st.empty()
    progress_bar = st.progress(0)
    result_placeholder = st.empty()

    real_url = get_real_audio_url(input_url)
    if not real_url:
        st.error("无法解析音频链接，请检查输入。")
        return

    session_id = uuid.uuid4().hex
    temp_source = f"src_{session_id}.m4a"

    try:
        status_box.info("1. 正在下载原始音频…")
        with requests.get(real_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_source, "wb") as f:
                for chunk in r.iter_content(1024 * 1024 * 2):
                    f.write(chunk)

        status_box.info("2. 正在切片（每片 10 分钟）…")
        chunk_pattern = f"chunk_{session_id}_%03d.mp3"
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                temp_source,
                "-f",
                "segment",
                "-segment_time",
                "600",
                "-c:a",
                "libmp3lame",
                "-ab",
                "64k",
                "-ar",
                "16000",
                "-ac",
                "1",
                chunk_pattern,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        chunk_files = sorted(
            f for f in os.listdir() if f.startswith(f"chunk_{session_id}_")
        )
        if not chunk_files:
            st.error("切片失败，可能是音频格式异常。")
            return

        full_text = ""
        total = len(chunk_files)

        for i, chunk in enumerate(chunk_files):
            status_box.info(f"3. 转写进度：{i + 1}/{total}")
            text = transcribe_with_retry(client, chunk)
            full_text += text + "\n"

            result_placeholder.text_area(
                label="实时结果",
                value=full_text,
                height=400,
            )

            os.remove(chunk)
            gc.collect()
            progress_bar.progress((i + 1) / total)

        status_box.success("✅ 转写完成！")
        st.download_button("下载完整文本", data=full_text, file_name="transcript.txt")
    except Exception as e:
        st.error(f"出错：{e}")
    finally:
        if os.path.exists(temp_source):
            os.remove(temp_source)
        for f in os.listdir():
            if f.startswith(f"chunk_{session_id}_"):
                try:
                    os.remove(f)
                except Exception:
                    pass

st.write("---")
url = st.text_input("请输入播客网页链接或音频直链")
if st.button("开始转写") and url:
    process_audio(url)