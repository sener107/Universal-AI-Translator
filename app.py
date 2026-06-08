import os

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

os.environ["ELEVENLABS_API_KEY"] = "a449604ef8f00145555104271f47b2b4adb72ed9bca0db002a587219258e093c"

import tempfile
import requests
import streamlit as st
import joblib
import torch

from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from streamlit_mic_recorder import mic_recorder


def get_elevenlabs_api_key():
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("ELEVENLABS_API_KEY", None)
        except Exception:
            api_key = None

    return api_key


voice_clone_enabled = get_elevenlabs_api_key() is not None


st.set_page_config(page_title="Universal Real-Time Translator V8A", page_icon="🌍")

st.title("🌍 Universal Real-Time Translator V8A")
st.caption(
    "Write, upload, speak, conversation mode or real-time chunks • "
    "Speech-to-text • Language detection • Translation • TTS • MP3 export"
)

if voice_clone_enabled:
    st.warning(
        "⚠️ ElevenLabs API key detected. Instant Voice Cloning requires a paid ElevenLabs plan."
    )
else:
    st.info("🔒 Voice Clone requires an ElevenLabs API key and paid plan.")


@st.cache_resource
def load_language_detector():
    language_model = joblib.load("fast_language_model_v2.pkl")
    le = joblib.load("label_encoder_v2.pkl")
    return language_model, le


@st.cache_resource
def load_translator():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_speech_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base"
    )


language_model, le = load_language_detector()
translator_tokenizer, translator_model = load_translator()
speech_model = load_speech_model()


nllb_codes = {
    "Turkish": "tur_Latn",
    "English": "eng_Latn",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Russian": "rus_Cyrl",
    "Arabic": "arb_Arab",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Hindi": "hin_Deva",
    "Thai": "tha_Thai",
    "Greek": "ell_Grek",
    "Czech": "ces_Latn",
    "Slovak": "slk_Latn",
    "Norwegian": "nob_Latn",
    "Dutch": "nld_Latn",
    "Swedish": "swe_Latn",
    "Ukrainian": "ukr_Cyrl",
    "Polish": "pol_Latn",
    "Albanian": "als_Latn",
    "Croatian": "hrv_Latn",
    "Serbian": "srp_Cyrl",
    "Azerbaijani": "azj_Latn",
    "Bosnian": "bos_Latn",
    "Bulgarian": "bul_Cyrl",
    "Estonian": "est_Latn",
    "Latvian": "lvs_Latn",
    "Armenian": "hye_Armn",
    "Persian": "pes_Arab",
    "Malay": "zsm_Latn",
    "Nepali": "npi_Deva",
    "Uzbek": "uzn_Latn",
    "Slovene": "slv_Latn",
    "Tibetan": "bod_Tibt",
    "Filipino": "tgl_Latn",
    "Urdu": "urd_Arab",
    "Danish": "dan_Latn",
    "Finnish": "fin_Latn",
    "Icelandic": "isl_Latn",
    "Irish": "gle_Latn",
    "Welsh": "cym_Latn",
    "Scottish Gaelic": "gla_Latn",
    "Catalan": "cat_Latn",
    "Galician": "glg_Latn",
    "Basque": "eus_Latn",
    "Belarusian": "bel_Cyrl",
    "Lithuanian": "lit_Latn",
    "Romanian": "ron_Latn",
    "Hungarian": "hun_Latn",
    "Hebrew": "heb_Hebr",
    "Pashto": "pbt_Arab",
    "Bengali": "ben_Beng",
    "Punjabi": "pan_Guru",
    "Gujarati": "guj_Gujr",
    "Marathi": "mar_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Odia": "ory_Orya",
    "Assamese": "asm_Beng",
    "Sanskrit": "san_Deva",
    "Vietnamese": "vie_Latn",
    "Indonesian": "ind_Latn",
    "Khmer": "khm_Khmr",
    "Lao": "lao_Laoo",
    "Burmese": "mya_Mymr",
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Igbo": "ibo_Latn",
    "Hausa": "hau_Latn",
    "Amharic": "amh_Ethi",
    "Somali": "som_Latn",
    "Zulu": "zul_Latn",
    "Xhosa": "xho_Latn",
    "Afrikaans": "afr_Latn"
}


tts_codes = {
    "Turkish": "tr",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Russian": "ru",
    "Arabic": "ar",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Italian": "it",
    "Portuguese": "pt",
    "Hindi": "hi",
    "Thai": "th",
    "Greek": "el",
    "Czech": "cs",
    "Slovak": "sk",
    "Norwegian": "no",
    "Dutch": "nl",
    "Swedish": "sv",
    "Ukrainian": "uk",
    "Polish": "pl",
    "Bulgarian": "bg",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Hebrew": "iw",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Afrikaans": "af",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Urdu": "ur"
}


whisper_language_map = {
    "Turkish": "turkish",
    "English": "english",
    "Spanish": "spanish",
    "French": "french",
    "German": "german",
    "Russian": "russian",
    "Arabic": "arabic",
    "Chinese": "chinese",
    "Japanese": "japanese",
    "Korean": "korean",
    "Italian": "italian",
    "Portuguese": "portuguese",
    "Hindi": "hindi",
    "Thai": "thai",
    "Greek": "greek",
    "Czech": "czech",
    "Slovak": "slovak",
    "Norwegian": "norwegian",
    "Dutch": "dutch",
    "Swedish": "swedish",
    "Ukrainian": "ukrainian",
    "Polish": "polish",
    "Bulgarian": "bulgarian",
    "Romanian": "romanian",
    "Hungarian": "hungarian",
    "Hebrew": "hebrew",
    "Persian": "persian",
    "Malay": "malay",
    "Urdu": "urdu",
    "Vietnamese": "vietnamese",
    "Indonesian": "indonesian",
    "Tamil": "tamil",
    "Telugu": "telugu",
    "Kannada": "kannada",
    "Malayalam": "malayalam",
    "Marathi": "marathi",
    "Gujarati": "gujarati",
    "Bengali": "bengali",
    "Afrikaans": "afrikaans"
}


def speech_to_text(audio_bytes, suffix=".wav", speech_language="Auto Detect"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
        fp.write(audio_bytes)
        audio_path = fp.name

    generate_kwargs = {"task": "transcribe"}

    if speech_language != "Auto Detect":
        whisper_language = whisper_language_map.get(speech_language)

        if whisper_language is not None:
            generate_kwargs["language"] = whisper_language

    result = speech_model(
        audio_path,
        generate_kwargs=generate_kwargs,
        return_timestamps=True
    )

    return result["text"]


def detect_language(text):
    prediction = language_model.predict([text])[0]
    language = le.inverse_transform([prediction])[0]
    return language


def translate_text(text, source_language, target_language):
    translator_tokenizer.src_lang = nllb_codes[source_language]

    inputs = translator_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        translated_tokens = translator_model.generate(
            **inputs,
            forced_bos_token_id=translator_tokenizer.convert_tokens_to_ids(
                nllb_codes[target_language]
            ),
            max_length=512
        )

    return translator_tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]


def create_standard_audio(text, language):
    tts_code = tts_codes.get(language)

    if tts_code is None:
        return None

    tts = gTTS(text=text, lang=tts_code)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name


def clone_speaker_voice(audio_bytes):
    api_key = get_elevenlabs_api_key()

    if not api_key:
        return None

    url = "https://api.elevenlabs.io/v1/voices/add"

    headers = {
        "xi-api-key": api_key
    }

    files = {
        "files": ("speaker_sample.wav", audio_bytes, "audio/wav")
    }

    data = {
        "name": "Universal Translator V7 Temporary Clone",
        "description": "Temporary cloned voice for Universal Translator V7"
    }

    response = requests.post(
        url,
        headers=headers,
        data=data,
        files=files,
        timeout=120
    )

    if response.status_code not in [200, 201]:
        st.warning(
            "Voice clone unavailable. Standard TTS will be used. "
            f"Detail: {response.text}"
        )
        return None

    return response.json().get("voice_id")


def create_elevenlabs_audio(text, voice_id):
    api_key = get_elevenlabs_api_key()

    if not api_key or not voice_id:
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.85
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        st.warning(
            "ElevenLabs TTS unavailable. Standard TTS will be used. "
            f"Detail: {response.text}"
        )
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        fp.write(response.content)
        return fp.name


def delete_elevenlabs_voice(voice_id):
    api_key = get_elevenlabs_api_key()

    if not api_key or not voice_id:
        return

    url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"

    headers = {
        "xi-api-key": api_key
    }

    try:
        requests.delete(url, headers=headers, timeout=60)
    except Exception:
        pass


def create_output_audio(
    translation,
    target_language,
    voice_output_mode="Standard TTS",
    reference_audio_bytes=None
):
    if (
        voice_output_mode == "Clone speaker voice"
        and reference_audio_bytes is not None
        and voice_clone_enabled
    ):
        voice_id = clone_speaker_voice(reference_audio_bytes)

        if voice_id is not None:
            cloned_audio = create_elevenlabs_audio(translation, voice_id)
            delete_elevenlabs_voice(voice_id)

            if cloned_audio is not None:
                return cloned_audio

    return create_standard_audio(translation, target_language)


def show_result(
    original_text,
    detected_language,
    target_language,
    translation,
    audio_file,
    download_name
):
    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Detected language: {detected_language}")

    with col2:
        st.info(f"Target language: {target_language}")

    st.subheader("📝 Original Text")
    st.text_area("Original:", original_text, height=120)

    st.subheader("🔄 Translation")
    st.text_area("Translated text:", translation, height=160)
    st.code(translation, language=None)

    if audio_file is not None:
        st.subheader("🔊 Spoken Translation")
        st.audio(audio_file)

        with open(audio_file, "rb") as file:
            st.download_button(
                label="Download MP3",
                data=file,
                file_name=download_name,
                mime="audio/mp3"
            )
    else:
        st.warning(f"Text-to-speech is not available for {target_language}.")


def run_pipeline(
    original_text,
    target_language,
    voice_output_mode="Standard TTS",
    reference_audio_bytes=None,
    download_name="translation_v7.mp3"
):
    detected_language = detect_language(original_text)

    if detected_language not in nllb_codes:
        st.error(f"Detected language is not supported: {detected_language}")
        return

    translation = translate_text(
        original_text,
        detected_language,
        target_language
    )

    audio_file = create_output_audio(
        translation,
        target_language,
        voice_output_mode,
        reference_audio_bytes
    )

    show_result(
        original_text,
        detected_language,
        target_language,
        translation,
        audio_file,
        download_name
    )

    return {
        "detected_language": detected_language,
        "target_language": target_language,
        "translation": translation
    }


def add_to_history(speaker, original, detected_language, target_language, translation):
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []

    st.session_state["conversation_history"].append(
        {
            "Speaker": speaker,
            "Original": original,
            "Detected Language": detected_language,
            "Target Language": target_language,
            "Translation": translation
        }
    )

if "realtime_history" not in st.session_state:
    st.session_state["realtime_history"] = []

input_mode = st.radio(
    "Input mode:",
    [
        "Write Text",
        "Upload Audio",
        "Speak with Microphone",
        "Conversation Mode",
        "Real-Time Mode"
    ]
)


if "last_input_mode" not in st.session_state:
    st.session_state["last_input_mode"] = input_mode

if st.session_state["last_input_mode"] != input_mode:
    st.session_state["final_text"] = ""
    st.session_state.pop("mic_audio_bytes", None)
    st.session_state.pop("conversation_audio_bytes", None)
    st.session_state["last_input_mode"] = input_mode


if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []


if input_mode in ["Write Text", "Upload Audio", "Speak with Microphone"]:

    speech_language = "Auto Detect"
    reference_audio_bytes = None
    voice_output_mode = "Standard TTS"

    if input_mode in ["Upload Audio", "Speak with Microphone"]:
        speech_language = st.selectbox(
            "Speech language:",
            ["Auto Detect"] + sorted(nllb_codes.keys())
        )

    if input_mode == "Speak with Microphone":
        voice_output_mode = st.radio(
            "Voice output:",
            ["Standard TTS", "Clone speaker voice"],
            horizontal=True
        )

    final_text = ""

    if input_mode == "Write Text":
        final_text = st.text_area("Write your text:")
        st.session_state["final_text"] = final_text

    elif input_mode == "Upload Audio":
        uploaded_audio = st.file_uploader(
            "Upload MP3, WAV or M4A:",
            type=["mp3", "wav", "m4a"],
            accept_multiple_files=False,
            key="audio_upload"
        )

        if uploaded_audio is not None:
            st.audio(uploaded_audio)

            if st.button("Convert Audio to Text"):
                with st.spinner("Converting audio to text..."):
                    suffix = "." + uploaded_audio.name.split(".")[-1]
                    final_text = speech_to_text(
                        uploaded_audio.read(),
                        suffix=suffix,
                        speech_language=speech_language
                    )
                    st.session_state["final_text"] = final_text

        final_text = st.session_state.get("final_text", "")

        if final_text:
            st.text_area("Detected text:", final_text, height=120)

    elif input_mode == "Speak with Microphone":
        st.write("Click record, speak, then stop.")

        audio = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            use_container_width=True,
            key="normal_mic_recorder_v7"
        )

        if audio is not None:
            st.session_state["mic_audio_bytes"] = audio["bytes"]
            st.audio(audio["bytes"], format="audio/wav")

        if "mic_audio_bytes" in st.session_state:
            reference_audio_bytes = st.session_state["mic_audio_bytes"]

            if st.button("Convert Speech to Text"):
                with st.spinner("Converting speech to text..."):
                    final_text = speech_to_text(
                        st.session_state["mic_audio_bytes"],
                        suffix=".wav",
                        speech_language=speech_language
                    )
                    st.session_state["final_text"] = final_text

        final_text = st.session_state.get("final_text", "")

        if final_text:
            st.text_area("Detected text:", final_text, height=120)

    target_language = st.selectbox(
        "Target language:",
        sorted(nllb_codes.keys()),
        index=sorted(nllb_codes.keys()).index("English")
    )

    if st.button("Detect, Translate & Speak"):
        final_text = st.session_state.get("final_text", final_text)

        if final_text.strip() == "":
            st.warning("Please provide text, record speech, or upload audio.")
        else:
            with st.spinner("Detecting, translating and creating speech..."):
                run_pipeline(
                    original_text=final_text,
                    target_language=target_language,
                    voice_output_mode=voice_output_mode,
                    reference_audio_bytes=reference_audio_bytes,
                    download_name="translation_v7.mp3"
                )


elif input_mode == "Conversation Mode":

    st.subheader("💬 Conversation Mode")

    col_a, col_b = st.columns(2)

    language_options = sorted(nllb_codes.keys())
    language_options_with_auto = ["Auto Detect"] + language_options

    with col_a:
        person_a_language = st.selectbox(
            "Person A Language:",
            language_options_with_auto
        )

    with col_b:
        person_b_language = st.selectbox(
            "Person B Language:",
            language_options,
            index=language_options.index("English")
        )

    direction = st.radio(
        "Direction:",
        ["A → B", "B → A"],
        horizontal=True
    )

    voice_output_mode = st.radio(
        "Voice output:",
        ["Standard TTS", "Clone speaker voice"],
        horizontal=True
    )

    if direction == "A → B":
        speaker = "A"
        speech_language = person_a_language
        target_language = person_b_language
    else:
        speaker = "B"
        speech_language = person_b_language
        target_language = person_a_language if person_a_language != "Auto Detect" else "Turkish"

    st.write(f"Current direction: **{direction}**")

    conversation_audio = mic_recorder(
        start_prompt="🎤 Record",
        stop_prompt="⏹️ Stop",
        just_once=True,
        use_container_width=True,
        key="conversation_mic_recorder_v7"
    )

    if conversation_audio is not None:
        st.session_state["conversation_audio_bytes"] = conversation_audio["bytes"]
        st.audio(conversation_audio["bytes"], format="audio/wav")

    if "conversation_audio_bytes" in st.session_state:
        if st.button("Transcribe & Translate Conversation"):
            with st.spinner("Transcribing, translating and generating speech..."):
                original_text = speech_to_text(
                    st.session_state["conversation_audio_bytes"],
                    suffix=".wav",
                    speech_language=speech_language
                )

                detected_language = detect_language(original_text)

                translation = translate_text(
                    original_text,
                    detected_language,
                    target_language
                )

                audio_file = create_output_audio(
                    translation,
                    target_language,
                    voice_output_mode,
                    st.session_state["conversation_audio_bytes"]
                )

                add_to_history(
                    speaker,
                    original_text,
                    detected_language,
                    target_language,
                    translation
                )

                show_result(
                    original_text,
                    detected_language,
                    target_language,
                    translation,
                    audio_file,
                    "conversation_translation_v7.mp3"
                )

    st.subheader("📜 Conversation History")

    if st.session_state["conversation_history"]:
        st.dataframe(
            st.session_state["conversation_history"],
            use_container_width=True
        )

        if st.button("Clear Conversation History"):
            st.session_state["conversation_history"] = []
            st.rerun()
    else:
        st.info("No conversation yet.")

elif input_mode == "Real-Time Mode":

    st.subheader("⚡ Real-Time Conversation Mode")
    st.write(
        "Use this mode for two-person live conversations. "
        "Person A speaks, the app translates to Person B. "
        "Then Person B speaks, the app translates back to Person A."
    )

    language_options = sorted(nllb_codes.keys())
    language_options_with_auto = ["Auto Detect"] + language_options

    col1, col2 = st.columns(2)

    with col1:
        person_a_language = st.selectbox(
            "Person A Language:",
            language_options_with_auto,
            index=0,
            key="rt_person_a_language"
        )

    with col2:
        person_b_language = st.selectbox(
            "Person B Language:",
            language_options,
            index=language_options.index("English"),
            key="rt_person_b_language"
        )

    voice_output_mode = st.radio(
        "Voice output:",
        ["Standard TTS", "Clone speaker voice"],
        horizontal=True,
        key="rt_voice_output"
    )

    st.info(
        "Record short turns. Example: A speaks → translate to B, "
        "then B speaks → translate to A."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🎤 Person A speaks")
        audio_a = mic_recorder(
            start_prompt="🎤 A Start",
            stop_prompt="⏹️ A Stop",
            just_once=True,
            use_container_width=True,
            key="rt_person_a_mic"
        )

        if audio_a is not None:
            st.session_state["rt_audio_a_bytes"] = audio_a["bytes"]
            st.audio(audio_a["bytes"], format="audio/wav")

        if "rt_audio_a_bytes" in st.session_state:
            if st.button("Translate A → B"):
                with st.spinner("Transcribing A and translating to B..."):
                    speech_language = person_a_language

                    original_text = speech_to_text(
                        st.session_state["rt_audio_a_bytes"],
                        suffix=".wav",
                        speech_language=speech_language
                    )

                    detected_language = detect_language(original_text)

                    target_language = person_b_language

                    translation = translate_text(
                        original_text,
                        detected_language,
                        target_language
                    )

                    audio_file = create_output_audio(
                        translation,
                        target_language,
                        voice_output_mode,
                        st.session_state["rt_audio_a_bytes"]
                    )

                    st.session_state["realtime_history"].append(
                        {
                            "Turn": "A → B",
                            "Speaker": "A",
                            "Original": original_text,
                            "Detected Language": detected_language,
                            "Target Language": target_language,
                            "Translation": translation
                        }
                    )

                    show_result(
                        original_text=original_text,
                        detected_language=detected_language,
                        target_language=target_language,
                        translation=translation,
                        audio_file=audio_file,
                        download_name="person_a_to_b_translation.mp3"
                    )

    with col_b:
        st.markdown("### 🎤 Person B speaks")
        audio_b = mic_recorder(
            start_prompt="🎤 B Start",
            stop_prompt="⏹️ B Stop",
            just_once=True,
            use_container_width=True,
            key="rt_person_b_mic"
        )

        if audio_b is not None:
            st.session_state["rt_audio_b_bytes"] = audio_b["bytes"]
            st.audio(audio_b["bytes"], format="audio/wav")

        if "rt_audio_b_bytes" in st.session_state:
            if st.button("Translate B → A"):
                with st.spinner("Transcribing B and translating to A..."):
                    speech_language = person_b_language

                    original_text = speech_to_text(
                        st.session_state["rt_audio_b_bytes"],
                        suffix=".wav",
                        speech_language=speech_language
                    )

                    detected_language = detect_language(original_text)

                    if person_a_language == "Auto Detect":
                        target_language = "Turkish"
                    else:
                        target_language = person_a_language

                    translation = translate_text(
                        original_text,
                        detected_language,
                        target_language
                    )

                    audio_file = create_output_audio(
                        translation,
                        target_language,
                        voice_output_mode,
                        st.session_state["rt_audio_b_bytes"]
                    )

                    st.session_state["realtime_history"].append(
                        {
                            "Turn": "B → A",
                            "Speaker": "B",
                            "Original": original_text,
                            "Detected Language": detected_language,
                            "Target Language": target_language,
                            "Translation": translation
                        }
                    )

                    show_result(
                        original_text=original_text,
                        detected_language=detected_language,
                        target_language=target_language,
                        translation=translation,
                        audio_file=audio_file,
                        download_name="person_b_to_a_translation.mp3"
                    )

    st.subheader("📡 Real-Time Conversation History")

    if st.session_state["realtime_history"]:
        st.dataframe(
            st.session_state["realtime_history"],
            use_container_width=True
        )

        full_original = "\n".join(
            f'{item["Turn"]}: {item["Original"]}'
            for item in st.session_state["realtime_history"]
        )

        full_translation = "\n".join(
            f'{item["Turn"]}: {item["Translation"]}'
            for item in st.session_state["realtime_history"]
        )

        st.subheader("📝 Full Conversation Transcript")
        st.text_area("Original conversation:", full_original, height=180)

        st.subheader("🔄 Full Conversation Translation")
        st.text_area("Translated conversation:", full_translation, height=180)

        st.download_button(
            "Download Conversation Transcript TXT",
            data=full_original,
            file_name="realtime_conversation_transcript_v8a.txt",
            mime="text/plain"
        )

        st.download_button(
            "Download Conversation Translation TXT",
            data=full_translation,
            file_name="realtime_conversation_translation_v8a.txt",
            mime="text/plain"
        )

        full_translation_audio = create_standard_audio(
            full_translation,
            person_b_language
        )

        full_original_audio = create_standard_audio(
    full_original,
    person_a_language if person_a_language != "Auto Detect" else "English"
)

        if full_original_audio is not None:
            with open(full_original_audio, "rb") as file:
                st.download_button(
                    "Download Full Original Conversation MP3",
                    data=file,
                    file_name="full_original_conversation.mp3",
                    mime="audio/mp3"
                )

        if full_translation_audio is not None:
            with open(full_translation_audio, "rb") as file:
                st.download_button(
                    "Download Full Conversation Translation MP3",
                    data=file,
                    file_name="full_conversation_translation.mp3",
                    mime="audio/mp3"
                )

        if st.button("Clear Real-Time Conversation History"):
            st.session_state["realtime_history"] = []
            st.rerun()

    else:
        st.info("No real-time conversation turns yet.")