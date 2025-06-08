import streamlit as st
import os
from mouse_detector import MouseDetector
import time
import pandas as pd
import cv2
from pathlib import Path
import base64

# Configure the page
st.set_page_config(page_title="Mouse Detector", layout="wide")

# Load custom font
def load_font():
    with open('static/fonts/ShareTechMono.otf', 'rb') as f:
        font_bytes = f.read()
    return base64.b64encode(font_bytes).decode()

# Load custom CSS with embedded font
def load_css():
    font_b64 = load_font()
    css = f'''
    @font-face {{
        font-family: 'ShareTechMono';
        src: url(data:font/otf;base64,{font_b64});
    }}

    /* Apply to everything */
    * {{
        font-family: 'ShareTechMono', monospace !important;
    }}

    /* Specific Streamlit elements */
    .stApp, .st-emotion-cache-*, div, p, span, li, td, th, 
    .stMarkdown, .stText, .stTextInput, .stTextArea, .stSelectbox, 
    .stMultiSelect, .stNumber, .stDataFrame, .stTable {{
        font-family: 'ShareTechMono', monospace !important;
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'ShareTechMono', monospace !important;
    }}

    /* Interactive elements */
    button, .stButton>button, .stSelectbox, 
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
        font-family: 'ShareTechMono', monospace !important;
    }}

    /* DataFrame elements */
    .dataframe td, .dataframe th {{
        font-family: 'ShareTechMono', monospace !important;
    }}
    '''
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Apply custom styling
load_css()

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'do_output_video' not in st.session_state:
    st.session_state.do_output_video = True
if 'do_plot_graphs' not in st.session_state:
    st.session_state.do_plot_graphs = True
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

st.title("Mouse Detector")

# File upload section
st.header("Загрузка видеофайла с экспериментом")

# Use a key for the file uploader to maintain its state
video_file = st.file_uploader("Загрузите видеофайл с экспериментом", type=['mp4', 'avi'], key='video_uploader')

# Handle file upload
if video_file is not None and (st.session_state.uploaded_file_name != video_file.name):
    # New file uploaded
    st.session_state.uploaded_file_name = video_file.name
    video_path = os.path.join("uploads", video_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    st.session_state.video_path = video_path
    st.session_state.analysis_complete = False  # Reset analysis state for new file
    st.success(f"Video uploaded successfully!")

# Analysis section
st.header("Запуск анализа")
col1, col2 = st.columns(2)

with col1:
    st.session_state.do_output_video = st.checkbox("Создать отладочное видео", value=st.session_state.do_output_video, key='debug_video_checkbox')
with col2:
    st.session_state.do_plot_graphs = st.checkbox("Создать графики анализа", value=st.session_state.do_plot_graphs, key='plots_checkbox')

def run_analysis():
    try:
        # Run detection
        mouse_detector = MouseDetector(
            st.session_state.video_path,
            "weights/weights_yolo.pt",
            "weights/behavior_weights_yolo.pt",
            st.session_state.do_output_video,
            st.session_state.do_plot_graphs
        )
        start = time.time()
        mouse_detector.detect()
        end = time.time()
        st.session_state.analysis_complete = True
        st.success(f"Анализ выполнен за {end - start:.2f} секунд(-ы)!")
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")

if st.button("Запуск", key='run_analysis_button'):
    if st.session_state.video_path is None:
        st.error("Сначала загрузите видеофайл с экспериментом!")
    else:
        with st.spinner("Анализируем видео..."):
            run_analysis()

# Results section - only show if analysis is complete
if st.session_state.analysis_complete and st.session_state.video_path:
    st.header("Результаты анализа")
    video_name = Path(st.session_state.video_path).stem
    
    # Excel data
    excel_path = f"mouse_data/{video_name}_data.xlsx"
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        st.subheader("Первые 20 строк данных")
        st.dataframe(df.head(20))
        
        col1, col2 = st.columns(2)
        with col1:
            with open(excel_path, "rb") as file:
                excel_data = file.read()
                st.download_button(
                    label="Скачать данные в Excel",
                    data=excel_data,
                    file_name=f"{video_name}_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='excel_download'
                )
    
    # CSV data
    csv_path = f"{video_name}_static.csv"
    if os.path.exists(csv_path):
        with col2:
            with open(csv_path, "rb") as file:
                csv_data = file.read()
                st.download_button(
                    label="Скачать данные в CSV",
                    data=csv_data,
                    file_name=f"{video_name}_raw_data.csv",
                    mime="text/csv",
                    key='csv_download'
                )

    # Display debug video if generated
    directory = os.path.dirname(st.session_state.video_path)
    filename = os.path.basename(st.session_state.video_path)
    processed_filename = f'processed_{filename}'
    debug_video_path = os.path.join(directory, processed_filename)
    if os.path.exists(debug_video_path):
        st.subheader("Отладочное видео")
        
        # Read video file
        with open(debug_video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            
        # Display video player
        st.video(video_bytes)
        
        # Add download button for video
        st.download_button(
            label="Скачать отладочное видео",
            data=video_bytes,
            file_name=f"processed_{Path(st.session_state.video_path).name}",
            mime="video/mp4",
            key='video_download'
        )

    # Display plots
    if st.session_state.do_plot_graphs:
        st.subheader("Графики анализа")
        plots_dir = f"mouse_data/{video_name}_plots"
        if os.path.exists(plots_dir):
            for plot_file in os.listdir(plots_dir):
                if plot_file.endswith(".png"):
                    st.image(
                        os.path.join(plots_dir, plot_file),
                        caption=plot_file.replace(".png", "").replace("_", " ").title(),
                        use_container_width=True
                    )

# Add some helpful information
st.sidebar.title("Краткая справка")

st.sidebar.write("Это приложение анализирует видеоэксперимент формата <<открытое поле>> над лабораторной мышью.")

st.sidebar.success("Поддерживаются видеофайлы в формате mp4 весом до 1Гб.")

st.sidebar.info("""
Алгоритм:
1. Загрузите видеофайл с экспериментом
2. Запустите анализ
3. Дождитесь результатов анализа
""")

st.sidebar.error("ВАЖНО: после получения результатов НЕ ОБНОВЛЯТЬ СТРАНИЦУ, иначе результаты будут потеряны!")

st.sidebar.subheader("Доступные функции:")
st.sidebar.write("""
- <<Создать отладочное видео>> - накладывает на исходный видеоролик найденные ключевые точки мыши и зоны арены, нужно для проверки работоспособности алгоритма
- <<Создать графики анализа>> - создает графики анализа на основе полученных данных.
- <<Скачать данные в Excel>> - скачивает данные анализа в формате Excel.
- <<Скачать данные в CSV>> - скачивает данные анализа в формате CSV.
- <<Скачать отладочное видео>> - скачивает отладочное видео.
""") 