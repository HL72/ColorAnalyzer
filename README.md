# 色調分析程式 (Color Tone Analysis)

https://coloranalyzer-kxdykucamsgwu7k7ztq88r.streamlit.app/
這是一個基於 Streamlit 建立的 Python 應用程式，用於分析和顯示圖片顏色資訊。

## Description

此程式接收圖片的顏色列表（`colors`），使用 K-Means Clustering 來對圖片中的所有像素進行分群後進行分析，並在介面為列表中的每種顏色提供詳細的資訊，包括其 HEX 色碼和通用的英文名稱。

## Features

* **圖片上傳**：支援 `.png`, `.jpg`, `.jpeg` 格式的圖片上傳。
* **可調色調數量**：使用者可透過側邊欄的滑桿，自由選擇要分析的顏色數量 (K值, 1-10種)。
* **視覺化圖表**：
    * **圓餅圖**
    * **長條圖**
* **顏色排序**：
    * 依百分比 (預設)：依佔比進行排序。
    * 依亮度 (由暗到亮)：根據顏色的感知亮度進行排序。
    * 依亮度 (由亮到暗)：同上，反向排序。
* **色碼資訊**：
    * 以色塊顯示圖片主要配色的 **HEX 碼**和**百分比**。
    * 在「詳細色碼」分頁中，提供 **HEX** 和 **RGB** 的色碼。
* **Palette**：
    * 產生包含所有主要顏色的Palette圖片。
    * 提供「下載 Palette」按鈕，方便使用者儲存該Palette。
* **透明度處理**：在分析過程中會自動過濾掉完全透明的像素 (Alpha < 0)，確保分析結果的準確性。

## Packages

要運行此程式，至少需要以下 Python 套件：

* **Streamlit**
* **Scikit-learn**
* **Pillow**
* **Matplotlib**
* **Pandas**
* **NumPy**
