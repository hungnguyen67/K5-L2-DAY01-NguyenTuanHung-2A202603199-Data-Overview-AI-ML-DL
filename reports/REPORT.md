# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K.
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Record này nói về **cả bức ảnh**, không phải một vật thể riêng trong ảnh. Vì mô hình phân loại ảnh dự đoán một lớp đại diện cho toàn bộ hình ảnh, không xác định vị trí từng vật thể.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Danh sách tên lớp được quy định sẵn và đi kèm với taxonomy/checkpoint. Model lựa chọn lớp trong danh sách đã được huấn luyện, không tự nghĩ ra tên lớp mới.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   Cần ghi cả ba thông tin vì:
   - `class_id`: mã số định danh của lớp.
   - `class_name`: tên lớp giúp con người dễ đọc và hiểu.
   - `taxonomy_name`: cho biết lớp thuộc hệ thống phân loại nào, tránh nhầm lẫn giữa các bộ nhãn khác nhau.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Guideline cần nói rõ nhãn phải mô tả chủ thể hay chủ đề đại diện cho toàn ảnh, cách chọn khi có nhiều loại vật thể và cách xử lý khi taxonomy không có lớp phù hợp. Như vậy annotator không tự chọn một vật thể riêng chỉ vì nó nổi bật, và các ảnh nhiều chủ đề được gán nhất quán. Ảnh `traffic` là cảnh giao thông nhưng dùng taxonomy ImageNet-1K, nên top-1 `cab` có thể trông lạ; điều đó tự nó chưa chứng minh model bị lỗi.

6. Vì sao không được lấy `score` của model làm ground truth?

   Không được lấy `score` của model làm ground truth vì `score` chỉ thể hiện mức độ tin cậy của dự đoán, không chứng minh dự đoán đó đúng. Ground truth phải được xác định dựa trên nội dung ảnh và guideline do con người thực hiện.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68.
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   Box có góc trên-trái tại tọa độ (385.33, 69.24) pixel và góc dưới-phải tại tọa độ (498.92, 348.92) pixel. Theo JSON, box rộng 113.58 pixel và cao 279.68 pixel. Hiệu hai tọa độ x đã làm tròn là 113.59; chênh lệch 0.01 pixel do JSON tính kích thước từ tọa độ gốc trước khi làm tròn tọa độ box. Gốc tọa độ (0, 0) nằm ở góc trên bên trái ảnh.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   Chọn ngưỡng 0.35 và 0.60: tại 0.35 có 11 predictions; tại 0.60 có 6 predictions. Số ở 0.20 cần lấy từ output notebook vì JSON hiện tại chỉ lưu kết quả với ngưỡng tạo file là 0.35.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   Reviewer cần kiểm tra các hộp có score thấp vì chúng có thể là vật thể bị nhận diện sai hoặc hộp không chính xác. Reviewer cần kiểm tra:
   - Hộp có thực sự bao quanh vật thể không.
   - Class được dự đoán có đúng không.
   - Có hộp trùng lặp cho cùng một vật thể không.
   - Có vật thể nào bị bỏ sót hoặc hộp nào bao gồm quá nhiều phần nền không.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   Quy tắc: Vẽ box bao sát phần vật thể nhìn thấy được, bao gồm toàn bộ phần nhìn thấy của vật thể nhưng không lấy quá nhiều nền hoặc phần của vật thể khác.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   Người ở sát mép trái ảnh bị khung hình cắt mất một phần cơ thể. Tôi sẽ đưa trường hợp này vào **guideline** vì cách xử lý có thể quy định rõ và áp dụng nhất quán: chỉ vẽ box quanh phần người nhìn thấy trong ảnh, cho phép box chạm mép ảnh và không suy đoán phần nằm ngoài khung hình.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[ [446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0] ].

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   Box chỉ cho biết hình chữ nhật bao quanh vật thể, còn polygon đi theo đường biên nên thể hiện hình dáng, chỗ lồi lõm và phần vật thể thực sự chiếm trong ảnh; nhờ vậy polygon thường loại bớt nền nằm trong box.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   `instance_id` phân biệt từng đối tượng riêng trong output này. Nó không biểu thị loại lớp (đó là `class_id`/`class_name`); hai người vẫn có thể cùng lớp nhưng có `instance_id` khác nhau. Nó cũng không theo dõi cùng một đối tượng qua nhiều khung hình, nên không phải tracking ID.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   Vẽ mask sát đường biên của phần vật thể nhìn thấy, không lấn sang nền hay vật thể bên cạnh; bỏ qua phần bị che khuất và không tự đoán đường biên ẩn.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   Ở góc dưới bên trái ảnh, bát màu hồng phía trước tiếp xúc với mặt bàn màu xanh; đường phân cách giữa hai mask khó xác định rõ. Tôi chọn **escalation** vì guideline chưa nêu cách xử lý biên của hai vật thể chạm nhau; cần hỏi Lab Coach để thống nhất thay vì tự suy đoán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

Lưu ý: workspace chỉ có prediction, không có ground truth do người gán nhãn tạo. Vì vậy, cột lỗi/chỗ khó dưới đây nêu rủi ro cần QC, không khẳng định model đã sai.

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                     | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                                                                                                     | Annotator làm gì?                                                                                                            | Reviewer xem gì?                                                                                              |
| --------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một `class_id`/`class_name` cho toàn ảnh theo taxonomy.           | Ảnh `traffic` có nhiều loại xe; JSON xếp `cab` hạng 1 (`score`=0.510915), `minibus` hạng 2 (0.164284). Cần guideline xác định lớp đại diện; score không xác nhận lớp đúng.                                             | Chọn lớp theo nội dung toàn ảnh và quy tắc chủ thể chính; hỏi khi guideline chưa phân định được.                             | Đối chiếu lớp với ảnh và guideline; kiểm tra taxonomy, mã/tên lớp và cách xử lý ảnh nhiều chủ đề.             |
| Phát hiện vật thể     | Mỗi vật thể: `class_id`/`class_name` và box `xyxy` theo pixel.    | PNG `detection_predictions.png` cho thấy người ở mép trái bị khung hình cắt; JSON có box person bắt đầu tại x=0.12. Đây là ca cần áp dụng nhất quán quy tắc vật thể bị cắt mép.                                        | Vẽ box quanh phần nhìn thấy, cho phép chạm mép ảnh; không suy đoán phần ngoài khung hình.                                    | Kiểm tra box có ôm phần nhìn thấy, đúng lớp; rà hộp trùng, bỏ sót và các hộp điểm thấp.                       |
| Instance segmentation | Mỗi vật thể: lớp, `instance_id` riêng và mask/polygon theo pixel. | PNG segmentation cho thấy bát màu hồng tiếp xúc với mặt bàn xanh ở góc dưới trái; đường phân cách hai mask khó xác định rõ tại vùng tiếp xúc. Đây là điểm mơ hồ, chưa thể kết luận polygon sai nếu thiếu ground truth. | Vẽ riêng từng instance theo phần biên nhìn thấy; escalation nếu guideline chưa rõ cách xử lý biên tiếp xúc hoặc phần bị che. | Kiểm tra mask bám biên, đúng instance/lớp, không lẫn vật thể; rà vùng tiếp xúc và phần bị che theo guideline. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Tôi chỉ sử dụng dữ liệu được cung cấp cho bài lab; không chia sẻ dữ liệu nhạy cảm và không ghi họ tên, MSSV, email hoặc số điện thoại vào báo cáo/output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   Tôi sẽ dừng xử lý, không tải lên hoặc chia sẻ dữ liệu đó, rồi báo Lab Coach hoặc giảng viên phụ trách để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md`
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
