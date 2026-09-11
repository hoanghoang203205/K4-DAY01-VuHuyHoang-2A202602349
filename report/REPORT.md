# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): "class_id": 468, "class_name": "cab", "rank": 1, "score": 0.510915,  "taxonomy_name": "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?

- Ai định nghĩa class list mà checkpoint có thể dự đoán?: Là những người gán nhãn dữ liệu 
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?: Bởi vì
    Taxonomy Xác định rõ hệ quy chiếu. Các bộ dữ liệu khác nhau có thể dùng chung một tên lớp nhưng mang ý nghĩa hoặc được đánh mã ID khác nhau.
    ID: Mã định danh dạng số duy nhất giúp máy tính xử lý nhanh, chính xác khi lập trình và không bị lỗi do sai lệch chuỗi ký tự
    Tên lớp: Dữ liệu dạng văn bản giúp con người đọc, hiểu và đánh giá ngay kết quả trực quan mà không cần phải tra cứu chéo với bảng mã
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    Nếu ảnh có nhiều chủ thể, guideline cần quy định rõ các tiêu chí cốt lõi sau:
        Tiêu chí xác định chủ thể chính: Cách chọn chủ thể đại diện cho toàn bộ ảnh (ví dụ: chiếm diện tích lớn nhất, nằm gần trung tâm nhất, hoặc rõ nét nhất) nếu mô hình chỉ xuất ra một nhãn duy nhất (single-label).

        Chính sách đa nhãn (Multi-label): Quy định rõ có cho phép gán nhiều nhãn trên cùng một ảnh hay không, và giới hạn số lượng nhãn tối đa được gán là bao nhiêu.
        Ngưỡng ghi nhận/bỏ qua: Định mức rõ khi nào một chủ thể phụ cần được gắn nhãn, và khi nào bị coi là "nhiễu" hoặc "bối cảnh" (background) để bỏ qua.

        Cách xử lý che khuất (Occlusion): Hướng dẫn phân loại trong trường hợp các chủ thể đè lấp lên nhau hoặc không lộ diện hoàn toàn khối hình.

- Vì sao model score không phải ground truth?: Vì đó là cái mà model dự đoán được. Còn ground truth là label để huấn luyện ai.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): "class_name": "bus",  "score": 0.912558, "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ], 
    "bbox_width": 129.84,
    "bbox_height": 132.96,

- Diễn giải vị trí box bằng lời:
    bbox_xyxy bao gồm 4 con số lần lượt là: xmin,ymin,xmax,ymax.
    "bbox_width": 35.22 Độ rộng của box
    "bbox_height": 28.42 Chiều cao của box
- So sánh số prediction ở hai threshold:
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?: 
    Sự thay đổi về độ bao phủ (coverage) và khối lượng công việc của reviewer phụ thuộc trực tiếp vào việc điều chỉnh ngưỡng tin cậy. 
    Khi tăng ngưỡng tin cậy: Độ bao phủ giảm do mô hình loại bỏ các dự đoán có độ tự tin thấp, kéo theo khối lượng dữ liệu mà reviewer cần xem xét thủ công sẽ giảm xuống.
    Khi giảm ngưỡng tin cậy: Độ bao phủ tăng do mô hình chấp nhận nhiều dự đoán hơn, kéo theo khối lượng dữ liệu reviewer cần xem xét sẽ tăng lên.
- Đề xuất một quy tắc box chặt:
    Bounding box phải được vẽ sao cho bao trọn vừa khít các điểm cực hạn (trên, dưới, trái, phải) của đối tượng.
    Không cắt lẹm (No cropping): Box phải chứa toàn bộ phần nhìn thấy được của vật thể, không cắt vào bất kỳ chi tiết chính nào.
    Không thừa nền (Minimum padding): Các cạnh của box phải chạm sát mép ngoài cùng của vật thể để thu hẹp tối đa không gian trống (background) bên trong.
    Loại trừ ngoại cảnh: Không bao gồm bóng đổ (shadows), hình phản chiếu dưới nước/kính, hoặc các phần bổ trợ quá nhỏ (như sợi dây mỏng, ăng-ten) trừ khi dự án có quy định bắt buộc.
    Xử lý che khuất (Occlusion): Nếu vật thể bị che khuất một phần, chỉ đóng box bao quanh phần thực tế còn nhìn thấy được, không nội suy phần bị che.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?: 
    Tỷ lệ hiển thị tối thiểu (Visibility Threshold): Vật thể cần hiển thị bao nhiêu phần trăm (ví dụ: >15% hay >30%) thì mới được phép dán nhãn, và dưới mức nào thì bỏ qua (Ignore).

    Nội suy hay không nội suy (Amodal vs. Modal): Chỉ đóng box đúng phần thực tế mắt nhìn thấy được (Modal), hay phải nội suy/ước lượng vẽ tràn ra để bao trọn cả phần bị che khuất hoặc phần bị cắt ngoài mép ảnh (Amodal).

    Gộp hay tách box (Merge vs. Split): Khi vật thể bị một chướng ngại vật (vd: thân cây, cột điện) che ngang và chia thành hai hoặc nhiều phần rời rạc, sẽ dùng một box lớn bao trùm tất cả hay dùng các box nhỏ dán riêng cho từng phần.

    Mất đặc trưng nhận diện: Nếu phần bị che/cắt chứa chi tiết quan trọng nhất để phân biệt class (ví dụ: đầu con vật, logo xe), có tiếp tục dán nhãn hay không.


## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): "instance_id": "traffic-001", "class_name": "bus",
    "score": 0.925745, "polygon_xy": [[148.0, 189.0],]
- Polygon bổ sung chi tiết gì so với box?: Polygon bổ sung thông tin polygon_point_count (số lượng điểm tạo nên đa giác) và mảng polygon_xy (tọa độ chi tiết của từng điểm trên viền mask), giúp xác định đường bao sát hình dáng thực tế của vật thể thay vì chỉ là một hình hộp chữ nhật.
- `instance_id` dùng để làm gì và không phải loại ID nào?: instance_id dùng để định danh duy nhất cho từng dự đoán segmentation cụ thể trong một ảnh (traffic-001, traffic-002). Nó khác với ID của ảnh hoặc ID của class 
- Đề xuất một quy tắc biên mask:Biên mask (polygon) phải bám sát mép ngoài cùng của vật thể, không lẹm vào trong (under-segmentation) và không tràn ra ngoài nền (over-segmentation).
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?: Cần quy định rõ hoặc đưa lên cấp cao hơn quyết định về việc ranh giới (boundary) sẽ được vẽ ở đâu khi ranh giới thực tế bị nhòe (motion blur/out-of-focus), hai vật thể cùng loại dính liền nhau khó phân biệt mép, hoặc khi vật thể bị che khuất thì mask có cần nội suy qua phần bị che không.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn class duy nhất đại diện cho toàn bộ bức ảnh. | Nhận diện sai logic (vd: bếp thành chiêng), hoặc mơ hồ khi ảnh có nhiều đối tượng đan xen (vd: nhà hàng và bàn ăn). | Quan sát tổng thể và gán một nhãn mô tả đúng nhất đối tượng hoặc ngữ cảnh chủ đạo. | Đánh giá tính đại diện của nhãn cho toàn ảnh và sự nhất quán giữa các class dễ nhầm lẫn. |
| Phát hiện vật thể | Hình hộp chữ nhật (tọa độ xyxy) kèm nhãn class. | Box lẹm chi tiết, thừa nền, sai class, hoặc khó xác định biên giới khi vật thể bị che khuất/cắt mép. | Vẽ bounding box bao trọn vừa khít phần nhìn thấy được của vật thể và gán nhãn. | Kiểm tra độ khít của box (không cắt lẹm, ít thừa nền nhất) và tính chính xác của nhãn. |
| Instance segmentation | Đa giác (tọa độ các điểm polygon) kèm nhãn class và ID độc lập (instance_id). | Mask vẽ lẹm hoặc tràn viền, khó xác định biên khi vật thể mờ nhòe hoặc dính liền nhau. | Vẽ đa giác bám sát viền thực tế của từng đối tượng ranh giới riêng biệt. | Đánh giá độ chính xác (pixel-perfect) của viền mask và đảm bảo phân tách đúng các vật thể |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không chia sẻ dữ liệu không liên quan và quan trọng ra bên ngoài.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Các bên có liên quan và thẩm quyền.

## 6. Danh sách bằng chứng

- [x ] `classification_predictions.json`
- [x ] `detection_predictions.json`
- [x ] `segmentation_predictions.json`
- [x ] `IMAGE_ATTRIBUTION.md`
- [x ] `visuals/classification_top5.png`
- [x ] `visuals/detection_predictions.png`
- [x ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [x ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
