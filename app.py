import gradio as gr
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typing import List, Tuple
import os
import gc 

# --- 1. Cấu hình và Tải Mô hình ---

# Đường dẫn đến mô hình đã lưu
MODEL_PATH = "./ner-biomedical-maccrobat2020-final"

# Tự động chọn thiết bị (GPU nếu có, nếu không thì CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang sử dụng thiết bị: {DEVICE}")

# Kiểm tra xem mô hình có tồn tại không
if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy thư mục mô hình tại '{MODEL_PATH}'.")
    print("Vui lòng đảm bảo mô hình đã được huấn luyện và lưu tại đúng vị trí.")
    exit()

try:
    # Tải tokenizer và mô hình đã được huấn luyện
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

    # Chuyển mô hình sang thiết bị đã chọn và đặt ở chế độ eval
    model.to(DEVICE)
    model.eval()
    print("Tải mô hình và tokenizer thành công.")

except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit()


# --- 2. Trích xuất Hàm Xử lý  ---

def inference(sentence: str, model, tokenizer, device: str = "cuda") -> Tuple[List[str], List[str]]:
    """
    Hàm inference gốc: Lấy câu, trả về token và nhãn dự đoán.
    """
    # 1. Tokenize input với tokenizer chuẩn (trả về input_ids + attention_mask)
    encoding = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 2. Dự đoán
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # 3. Lấy nhãn dự đoán (argmax)
    preds = torch.argmax(logits, dim=-1).squeeze(0)  # [seq_len]

    # 4. Map ids → labels
    preds_labels = [model.config.id2label[p.item()] for p in preds]

    # 5. Lấy token đã tokenize
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    return tokens, preds_labels


def merge_entity(tokens: List[str], preds_labels: List[str]) -> List[Tuple[str, str]]:
    """
    Hàm merge_entity gốc: Nhóm các token và nhãn B-I-O thành các thực thể.
    Ví dụ: ("head", "B-Symptom"), ("##ache", "I-Symptom") -> ("Symptom", "head ache")
    """
    merged_list = []
    temp_tokens = []
    current_label = None

    for token, label in zip(tokens, preds_labels):
        # Bỏ qua các token đặc biệt
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue

        # Lấy type thực sự (bỏ B-/I-), giữ O
        type_label = label.split("-")[-1]

        if type_label == "O":
            # Nếu là 'O', lưu thực thể cũ (nếu có)
            if temp_tokens:
                merged_list.append((current_label, " ".join(temp_tokens).replace(" ##", "")))
                temp_tokens = []
                current_label = None
            # Thêm token 'O' vào
            merged_list.append((type_label, token.replace("##", "")))
        else:
            # Nếu là một thực thể (B- hoặc I-)
            if current_label == type_label:
                # Nếu cùng loại (ví dụ: I-Symptom tiếp B-Symptom), thêm token
                temp_tokens.append(token)
            else:
                # Nếu là nhãn mới, lưu thực thể cũ (nếu có)
                if temp_tokens:
                    merged_list.append((current_label, " ".join(temp_tokens).replace(" ##", "")))
                # Bắt đầu thực thể mới
                temp_tokens = [token]
                current_label = type_label

    # Lưu thực thể cuối cùng còn sót lại
    if temp_tokens:
        merged_list.append((current_label, " ".join(temp_tokens).replace(" ##", "")))

    return merged_list


# --- 3. Hàm Predict Chính cho Gradio ---

def predict_entities(text: str) -> Tuple[list, str]:
    """
    Hàm chính được gọi bởi Gradio Interface.
    """
    if not text:
        gc.collect() 
        return [], "Vui lòng nhập văn bản."

    # 1. Chạy inference
    tokens, preds_labels = inference(text, model, tokenizer, device=DEVICE)

    # 2. Xử lý hậu kỳ (merge subwords và entities)
    merged_results = merge_entity(tokens, preds_labels)

    # 3. Định dạng cho 2 output:
    
    # Output 1: Dành cho gr.HighlightedText
    # Cần định dạng: [(text, label), (text, None), (text, label)]
    highlight_output = []
    for label, text in merged_results:
        if label == "O":
            highlight_output.append((text, None))
        else:
            highlight_output.append((text, label))

    # Output 2: Dành cho gr.Textbox (danh sách)
    # Lọc ra các thực thể (bỏ 'O')
    entities_only = [f"• {text.strip()} ({label})" for label, text in merged_results if label != "O"]
    
    if not entities_only:
        text_output = "Không tìm thấy thực thể y khoa nào."
    else:
        text_output = "Các thực thể tìm thấy:\n" + "\n".join(entities_only)

    gc.collect()

    return highlight_output, text_output

# --- 4. Xây dựng Giao diện Gradio ---

APP_TITLE = "Trợ lý Phân tích Y khoa (Medical NER)"
DESCRIPTION_MD = """
# 🩺 Trợ lý Phân tích Văn bản Y khoa
Chào mừng đến với công cụ Trích xuất Thực thể Y khoa (NER), được thiết kế để hỗ trợ các chuyên gia y tế và nhà nghiên cứu.

**Mục tiêu:** Tự động xác định và phân loại các thông tin lâm sàng quan trọng từ văn bản y khoa.
**Cách dùng:** Dán một đoạn văn bản (ví dụ: bệnh án, tóm tắt ca bệnh) vào ô bên dưới.

Hệ thống sẽ tự động phân tích và highlight các thực thể bao gồm:
- **Bệnh lý/Rối loạn:** `Disease_disorder`
- **Dấu hiệu/Triệu chứng:** `Sign_symptom`
- **Cấu trúc sinh học:** `Biological_structure`
- **Thủ thuật điều trị:** `Therapeutic_procedure`
- **Thủ thuật chẩn đoán:** `Diagnostic_procedure`
- **Thông tin nhân khẩu học:** `Age`, `Sex`
"""


EXAMPLE_TEXT = ("A 48 year - old female presented with vaginal bleeding and abnormal Pap smears . "
              "Upon diagnosis of invasive non - keratinizing SCC of the cervix , "
              "she underwent a radical hysterectomy with salpingo - oophorectomy "
              "which demonstrated positive spread to the pelvic lymph nodes and the parametrium . "
              "Pathological examination revealed that the tumour also extensively involved the lower uterine segment .")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION_MD)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Văn bản Y khoa",
                placeholder="Nhập một đoạn văn bản y khoa tại đây...",
                lines=10,
                value=EXAMPLE_TEXT 
            )
            submit_btn = gr.Button("🔬 Phân tích Thực thể", variant="primary")
            
        with gr.Column(scale=3):
            gr.Markdown("#### Kết quả Phân tích Trực quan") 
            highlight_output = gr.HighlightedText(
                label="Phân tích NER",
                show_legend=True, 
                color_map={ 
                    "Sign_symptom": "pink",
                    "Disease_disorder": "red",
                    "Biological_structure": "green",
                    "Age": "gray",
                    "Sex": "gray",
                    "Therapeutic_procedure": "blue",
                    "Diagnostic_procedure": "purple",
                    "Lab_value": "orange",
                }
            )
            gr.Markdown("#### Tóm tắt Thực thể") 
            text_output = gr.Textbox(label="Danh sách (chỉ đọc)", interactive=False, lines=7)

    # Liên kết nút bấm với hàm xử lý
    submit_btn.click(
        fn=predict_entities,
        inputs=text_input,
        outputs=[highlight_output, text_output]
    )
    
    # Thêm ví dụ cho người dùng
    gr.Examples(
        examples=[
            [EXAMPLE_TEXT],
            ["The patient reported persistent headaches and blurred vision."],
            ["CT scan revealed a 2 cm mass in the left lung lobe."]
        ],
        inputs=text_input,
        outputs=[highlight_output, text_output],
        fn=predict_entities,
        cache_examples=True 
    )

# --- 5. Chạy Ứng dụng ---
if __name__ == "__main__":
    demo.launch(debug=True)