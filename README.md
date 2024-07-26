# test
 網址:https://github.com/Chen-Yen-Kai/Chest-X-ray-Images-Using-Stacked-Ensemble-Learning
 主程式檔案AutomaticAugmentationTraining，包含擴增訓練，此程式為單一網路
 結果會存取再選擇的xlse檔案裏
 訓練好的模型，在ensemble檔案可以選擇集成模型
 ensemble檔案裏面的ensemble_LR為邏輯回歸
 ensemble_deselayer為deselayer方法，結果會存取在lr.ALL.xls
 
如何製作label
使用Photoshop

標記在原始X光影像上的情況(成大)
1.放入標記影像
2.建立新圖層
3.按住Alt並於圖層處雙擊原始X光影像，使其為可編輯狀態
4.進入新圖層並填滿黑色(前景與背景)
5.使用快速選取工具圈選病灶範圍
6.進入新圖層並填滿像素值(前景與背景)，依照類別一填入1，類別二填入2，以此類推
7.儲存為PNG檔


label圖與原始X光影像分開的情況(深圳)
1. 放入label圖
2. 建立圖層並放上X光PNG檔
3. 影像-模式-RGB (點陣化，不要合併)
4. 自動選取病灶範圍
5. 選擇前景色
6. 切到label圖-快速選取
7. 切到X光影像-在病灶範圍上按右鍵-筆畫(大小4)-取消選取
8. 存檔

*DICOM影像轉PNG(DICOM_to_PNG資料夾)
使用DICOM_to_PNG.py
DICOM影像放在DICOM資料夾中



*影像前處理(image_preprocessing資料夾)

saved_models資料夾放入肺部分割與骨骼抑制的權重檔

使用run.py
放入PNG影像至test資料夾，醫師繪製mask影像放置result\mask
result\label顯示醫師繪製mask影像的ROI區域
result\box_result顯示ROI範圍
result\image_or_size顯示裁切後的ROI影像
result\mask_image_area顯示肺部分割的影像

使用BS.py
將result\image_or_size與result\mask_image_area中的影像加入骨骼抑制
result\image_or_size_bs顯示骨骼抑制後的ROI影像
result\mask_image_area_bs顯示骨骼抑制後的肺部分割的影像



*k_fold(2class_k_fold資料夾)
內容與影像擴增訓練一樣，增加k_fold部分
使用fit_k_fold.py
將所有影像放入image資料中，label放入label資料夾中，醫師標記影像放入mark資料夾中




*影像擴增訓練(2class_MODEL資料夾)
四種不同前處理影像資料夾




*現有模型測試或擴增後圖片訓練(2class_MODEL資料夾)
模型測試使用ReadModel.py，程式內修改要測試的資料夾名稱、模型名稱、損失函數名稱等各資訊
擴增後圖片訓練使用ReadAugmentationTraining.py



*集成式網路

