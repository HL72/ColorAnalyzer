import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
from collections import Counter
from io import BytesIO
import pandas as pd



#分析函數
def analyze_colors(image_data, k):
    img = image_data.convert('RGBA')
    img_np = np.array(img)
    pixels_rgba = img_np.reshape(-1, 4)
    alpha_channel = pixels_rgba[:, 3]
    opaque_mask = alpha_channel > 0
    pixels_rgb_opaque = pixels_rgba[opaque_mask, :3]

    #特殊情況：圖片是完全透明的
    if pixels_rgb_opaque.shape[0] == 0:
        st.warning("這是一張完全透明的圖片，無法分析顏色。")
        return [], []

    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(pixels_rgb_opaque)

    colors = kmeans.cluster_centers_.astype(int)
    counts = Counter(kmeans.labels_)

    total_pixels = len(kmeans.labels_)
    percentages = [counts[i] / total_pixels for i in range(k)]

    combined = sorted(zip(colors, percentages), key=lambda x: x[1], reverse=True)
    sorted_colors = [item[0] for item in combined]
    sorted_percentages = [item[1] for item in combined]

    return sorted_colors, sorted_percentages

#圓餅圖函數
def create_pie_chart(colors, percentages):
    colors_normalized = [tuple(c / 255.0) for c in colors]
    fig, ax = plt.subplots(figsize=(6, 6))

    wedges, texts, autotexts = ax.pie(
        percentages,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_normalized,
        wedgeprops={'edgecolor': 'white'}
    )

    for i, autotext in enumerate(autotexts):
        text_color = 'white' if sum(colors[i]) < 384 else 'black'
        autotext.set_color(text_color)
        autotext.set_fontweight('bold')

    ax.axis('equal')
    plt.tight_layout()
    return fig

#Palette函數
def create_palette_image(colors, percentages, height=100):
    width = 600
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    num_colors = len(colors)

    if num_colors == 0:
        return img

    block_width = width // num_colors
    start_x = 0

    for i, col in enumerate(colors):
        end_x = start_x + block_width
        if i == num_colors - 1:
            end_x = width
        draw.rectangle([start_x, 0, end_x, height], fill=tuple(map(int, col)))
        start_x = end_x

    return img

#RGB轉HEX函數
def rgb_to_hex(rgb):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return f'#{r:02x}{g:02x}{b:02x}'

#計算亮度函數
def get_brightness(rgb):
    r, g, b = rgb
    #Luma (Y') formula: Y' = 0.299R + 0.587G + 0.114B
    return 0.299 * r + 0.587 * g + 0.114 * b

#標題
st.set_page_config(layout="wide")
st.title("圖片色調分析器")
st.write("請上傳一張圖片，程式將使用演算法分析圖片的主要配色。")

#側欄
st.sidebar.header("設定")
k_slider = st.sidebar.slider("請選擇要分析的色調數量", min_value=1, max_value=10, value=5)
st.sidebar.subheader("顯示選項")
chart_type = st.sidebar.radio(
    "圖表類型",
    options=["圓餅圖", "長條圖"],
    index=0,
    horizontal=True
)
sort_by = st.sidebar.selectbox(
    "顏色排序方式",
    options=["依百分比 (預設)", "依亮度 (由暗到亮)", "依亮度 (由亮到暗)"],
    index=0
)
st.sidebar.markdown("---")
with st.sidebar.expander("關於此程式"):
    st.markdown("此程式使用 **K-Means Clustering** 來對圖片中的所有像素進行分群。")

#主畫面
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        display_image = image.convert('RGB')
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(display_image, caption="原始圖片", use_container_width=True)

        with col2:
            st.subheader(f"分析結果：")
            with st.spinner("正在分析中..."):
                colors, percentages = analyze_colors(image, k_slider)

            if colors:
                combined_results = list(zip(colors, percentages))

                if sort_by == "依亮度 (由暗到亮)":
                    combined_results.sort(key=lambda x: get_brightness(x[0]))
                elif sort_by == "依亮度 (由亮到暗)":
                    combined_results.sort(key=lambda x: get_brightness(x[0]), reverse=True)

                colors = [item[0] for item in combined_results]
                percentages = [item[1] for item in combined_results]

            if colors:
                tab1, tab2, tab3 = st.tabs(["顏色佔比", "色碼百分比、Palette", "詳細色碼"])

                with tab1:
                    st.subheader("主要顏色佔比")

                    if chart_type == "圓餅圖":
                        fig = create_pie_chart(colors, percentages)
                        st.pyplot(fig)

                    elif chart_type == "長條圖":

                        hex_colors = [rgb_to_hex(c) for c in colors]

                        df_data = {
                            "Percentage": percentages,
                            "ColorHex": hex_colors,
                        }
                        df = pd.DataFrame(df_data, index=hex_colors)

                        st.write("由佔比高至低排序（或依您選擇的排序方式）")

                        st.bar_chart(
                            df,
                            y="Percentage",
                            color="ColorHex",
                            use_container_width=True
                        )
                        st.caption("提示：將滑鼠懸停在長條上可查看精確百分比。")

                with tab2:
                    st.write("**色碼百分比**")

                    num_colors = len(colors)
                    num_cols = min(num_colors, 5)

                    if num_colors > 0:
                        cols = st.columns(num_cols)
                        for i in range(num_colors):
                            with cols[i % num_cols]:
                                rgb = colors[i]
                                hex_color = rgb_to_hex(rgb)
                                percentage_str = f"{percentages[i]:.1%}"
                                text_color = 'white' if sum(rgb) < 384 else 'black'

                                st.markdown(f"""
                                    <div style="
                                        background-color: {hex_color};
                                        color: {text_color};
                                        padding: 20px 10px;
                                        border-radius: 8px;
                                        text-align: center;
                                        font-family: monospace;
                                        font-weight: bold;
                                        line-height: 1.6;
                                        margin-bottom: 15px;
                                    ">
                                        {hex_color.upper()}<br>
                                        {percentage_str}
                                    </div>
                                    """, unsafe_allow_html=True)

                    st.markdown(f"---")
                    st.write("")

                    pal_img = create_palette_image(colors, percentages, height=80)
                    st.image(pal_img, use_container_width=False)

                    buf = BytesIO()
                    pal_img.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button("下載 Palette", data=buf, file_name="Palette.png", mime="image/png")

                with tab3:
                    for i in range(len(colors)):
                        rgb = colors[i]
                        hex_color = rgb_to_hex(rgb).upper()
                        rgb_str = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

                        st.markdown(f"--- \n #### {i + 1}. {hex_color.upper()} ({percentages[i]:.1%})")

                        st.markdown(f"""
                                <div style="width:100%; height: 40px; background-color:{hex_color}; border-radius: 5px; border: 1px solid #ddd;"></div>
                                """, unsafe_allow_html=True)
                        st.write("")

                        st.write("**HEX**")
                        st.code(hex_color, language="css")
                        st.write("**RGB**")
                        st.code(rgb_str, language="css")

            elif not st.session_state.get('warning_shown', False):
                pass

    except Exception as e:
        st.error(f"分析失敗，請確認圖片格式是否正確。錯誤訊息：{e}")

else:
    st.info("請上傳一張圖片以開始分析。")