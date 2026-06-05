"""
Chuẩn hóa 4 dataset chính:
- intent_record.csv, intent_action.csv, intent_chitchat.csv, ner_dataset.jsonl

Bước:
1) Dedupe theo ``text`` (giảm trùng làm TF-IDF nặng một vài mẫu).
2) Gộp nhãn mâu thuẫn: action_type / sentiment lấy mode; ``SYSTEM_SETTING`` → ``Setting``.
3) Bổ sung mẫu biên (Record vs Action vs Chitchat) + hard negative UI + sentiment rõ + NER.

Chạy: python improve_datasets.py

Sau đó (nếu thêm nhiều mẫu intent / action / chitchat):
  python text_nlu/train/retrain_encoders.py
  python text_nlu/train/train_category_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RECORD_CSV = ROOT / "intent_record.csv"
ACTION_CSV = ROOT / "intent_action.csv"
CHITCHAT_CSV = ROOT / "intent_chitchat.csv"
NER_JSONL = ROOT / "ner_dataset.jsonl"


def _mode(series: pd.Series) -> str:
    m = series.mode()
    return str(m.iloc[0]) if len(m) else str(series.iloc[0])


def dedupe_record(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)


def dedupe_action(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("text", as_index=False)
        .agg(intent=("intent", "first"), action_type=("action_type", _mode))
        .reset_index(drop=True)
    )


def dedupe_chitchat(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("text", as_index=False)
        .agg(intent=("intent", "first"), sentiment=("sentiment", _mode))
        .reset_index(drop=True)
    )


def normalize_action_types_column(df: pd.DataFrame) -> pd.DataFrame:
    """Gộp nhãn hiếm / trùng nghĩa để encoder action_type ổn định."""
    out = df.copy()
    out["action_type"] = out["action_type"].replace({"SYSTEM_SETTING": "Setting"})
    return out


def dedupe_ner_lines(lines: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in lines:
        t = row["text"].strip()
        if t in seen:
            continue
        seen.add(t)
        out.append(row)
    return out


def augment_chitchat(existing: set[str]) -> list[dict]:
    rows: list[dict] = []
    openings = [
        "Chào",
        "Hi",
        "Ê",
        "Nè",
        "Hế lô",
        "Yo",
        "Hello",
        "Chao ban",
        "chao bot",
        "hom nay",
        "mai minh",
        "toi buon",
        "met qua di",
        "vui qua ha",
        "cam on nhe",
        "thanks ban",
        "ban ten gi",
        "ban la ai",
        "app nay de lam gi",
        "cho minh hoi nhanh",
        "khong lien quan tien",
        "noi choi thoi",
        "doi luc met",
        "sao ban im lang",
        "ket thuc chat nhe",
        "hen gap lai",
        "bye nha",
        "ngu ngon",
        "chuc ngon mieng",
        "thoi tiet the nao",
        "ban khoe khong",
        "minh hoi chut",
        "khong biet hoi ai",
        "tam biet",
        "see you",
        "okela",
        "uhm uhm",
        "hihi",
        "haha",
        "tro ly thong minh qua",
        "ban noi hay qua",
        "minh chi tam su",
        "doi luc can dong vien",
        "hom nay met qua roi",
        "ke chuyen vui di",
        "noi chuyen chan thoi",
        "khong muon nhap chi tieu",
        "chi muon tam su",
        "ban co ban khong",
        "gio may roi",
        "hom nay la thu may",
        "ban noi tieng Viet a",
        "test chat thoi",
        "ping bot",
        "co ai khong",
        "rảnh không",
        "tam dung chut",
    ]
    tails = [
        " nha",
        " di",
        " hen",
        " vay",
        " thoi",
        " a",
        " oi",
        " bot",
        " nhe",
        " ha",
        "",
        " may",
        " ban",
        " minh di day",
        " chut xiu",
        " thui",
        " thoi nha",
    ]
    def sentiment_for_line(full: str) -> str:
        t = full.lower()
        if any(k in t for k in ("cam on", "thanks", "vui", "hihi", "haha", "thich", "hay qua", "ngon mieng", "tot ", "yeu ")):
            return "Positive"
        if any(k in t for k in ("buon", "met ", "met qua", "chan ", "that vong", "stress", "that buon")):
            return "Negative"
        return "Neutral"

    for o in openings:
        for tail in tails:
            text = (o + tail).strip()
            if len(text) < 3 or text in existing:
                continue
            existing.add(text)
            sent = sentiment_for_line(text)
            rows.append({"text": text, "intent": "Chitchat", "sentiment": sent})
            if len(rows) >= 900:
                return rows
    return rows


def augment_action(existing: set[str]) -> list[dict]:
    """Mẫu Action đa dạng; ``action_type`` khớp nhãn đang dùng trong CSV."""
    specs: list[tuple[str, str]] = [
        ("Thống kê {cat} tháng này", "Report"),
        ("Báo cáo {cat} tuần trước", "REPORT_GENERAL"),
        ("So sánh {cat} tuần này với tuần trước", "REPORT_COMPARE"),
        ("Tìm khoản {cat} trên {amt}", "SEARCH_RECORD"),
        ("Xóa giao dịch {cat} vừa nhập", "DELETE_RECORD"),
        ("Đặt hạn mức {cat} {amt}", "SET_LIMIT"),
        ("Giảm hạn mức {cat} xuống {amt}", "SET_LIMIT"),
        ("Tăng mục tiêu {cat} lên {amt}", "ADD_GOAL"),
        ("Đặt mục tiêu tiết kiệm {amt}", "SET_GOAL"),
        ("Lọc giao dịch {cat} trong tháng", "SEARCH_RECORD"),
        ("Cho xem biểu đồ {cat}", "Report"),
        ("Xuất dữ liệu {cat} ra Excel", "EXPORT_DATA"),
        ("Đổi giọng bot sang hài hước", "SET_TONE"),
        ("Đặt cảnh báo khi chi {cat} quá {amt}", "SET_ALERT"),
        ("Cập nhật khoản {cat} thành {amt}", "UPDATE_RECORD"),
        ("Đặt thu nhập cố định {amt}", "SET_INCOME"),
        ("Đổi tên hiển thị thành Minh Anh", "SET_USERNAME"),
        ("Tắt đồng bộ dữ liệu đám mây", "Setting"),
        ("tim kiem khoan {cat} lon hon {amt}", "SEARCH_RECORD"),
        ("bao cao tong chi thang nay", "REPORT_GENERAL"),
        ("xoa ban ghi gan nhat", "DELETE_RECORD"),
        ("sua lai khoan {cat} thanh {amt}", "UPDATE_RECORD"),
    ]
    cats = [
        "ăn uống",
        "đi lại",
        "mua sắm",
        "giải trí",
        "điện nước",
        "học phí",
        "y tế",
        "nhà cửa",
    ]
    amts = ["100k", "200k", "1tr", "2tr", "500k", "50k", "300k", "1.5tr"]
    rows: list[dict] = []
    for tpl, atype in specs:
        has_cat = "{cat}" in tpl
        has_amt = "{amt}" in tpl
        if not has_cat and not has_amt:
            text = tpl.strip()
            if text not in existing:
                existing.add(text)
                rows.append({"text": text, "intent": "Action", "action_type": atype})
            if len(rows) >= 480:
                return rows
            continue
        if has_cat and has_amt:
            combos = [(c, a) for c in cats for a in amts]
        elif has_cat:
            combos = [(c, None) for c in cats]
        else:
            combos = [(None, a) for a in amts]
        for cat, amt in combos:
            if has_cat and has_amt:
                text = tpl.format(cat=cat, amt=amt).strip()
            elif has_cat:
                text = tpl.format(cat=cat).strip()
            else:
                text = tpl.format(amt=amt).strip()
            if text in existing:
                continue
            existing.add(text)
            rows.append({"text": text, "intent": "Action", "action_type": atype})
            if len(rows) >= 480:
                return rows
    return rows


def augment_record(existing: set[str]) -> list[dict]:
    """Mẫu Record rõ ràng (có tiền) — tránh nhầm Action/Chitchat."""
    labels = ["Food", "Transport", "Shopping", "Entertainment", "Housing", "Others"]
    items = [
        "cơm trưa văn phòng",
        "xăng xe máy",
        "vé xe buýt",
        "bánh mì sáng",
        "trà sữa",
        "Netflix",
        "Spotify",
        "gửi xe",
        "wifi nhà",
        "thuốc cảm",
    ]
    amts = ["25k", "35k", "48k", "60k", "120k", "199k", "250k", "500k"]
    rows: list[dict] = []
    for lab in labels:
        for it in items:
            for a in amts:
                text = f"{it} hết {a} (nhóm {lab})"
                if text in existing:
                    continue
                existing.add(text)
                rows.append({"text": text, "label": lab, "type": "expense", "is_money": 1})
                if len(rows) >= 180:
                    return rows
    return rows


def ner_spans(text: str, spans: list[tuple[int, int, str]]) -> dict:
    return {"text": text, "label": [[s, e, lb] for s, e, lb in spans]}


def augment_ner(existing: set[str]) -> list[dict]:
    """Thêm NER: subscription + không dấu + action."""
    rows: list[dict] = []

    def add(text: str, parts: list[tuple[str, str]]):
        """parts: (substring, label) — tìm vị trí đầu tiên."""
        if text in existing:
            return
        spans: list[tuple[int, int, str]] = []
        for sub, lb in parts:
            i = text.find(sub)
            if i < 0:
                return
            spans.append((i, i + len(sub), lb))
        existing.add(text)
        rows.append(ner_spans(text, spans))

    pairs = [
        ("thanh toán Netflix tháng 109k", [("Netflix", "CATEGORY"), ("109k", "AMOUNT")]),
        ("gia hạn Spotify 59k", [("Spotify", "CATEGORY"), ("59k", "AMOUNT")]),
        ("mua gói YouTube Premium 79k", [("YouTube Premium", "CATEGORY"), ("79k", "AMOUNT")]),
        ("chi phí ChatGPT Plus 20 đô", [("ChatGPT Plus", "CATEGORY"), ("20 đô", "AMOUNT")]),
        ("tiền điện tháng 450k", [("điện", "CATEGORY"), ("450k", "AMOUNT")]),
        ("tìm kiếm khoản ăn uống trên 200k", [("tìm kiếm", "ACTION_TYPE"), ("ăn uống", "CATEGORY"), ("200k", "AMOUNT")]),
        ("thống kê đi lại tuần trước", [("thống kê", "ACTION_TYPE"), ("đi lại", "CATEGORY"), ("tuần trước", "TIME")]),
        ("so sánh mua sắm hôm nay và hôm qua", [("so sánh", "ACTION_TYPE"), ("mua sắm", "CATEGORY"), ("hôm nay", "TIME"), ("hôm qua", "TIME")]),
        ("xóa giao dịch 35k vừa rồi", [("xóa", "ACTION_TYPE"), ("35k", "AMOUNT")]),
        ("đặt hạn mức giải trí 1tr", [("đặt", "VERB"), ("hạn mức", "TARGET"), ("giải trí", "CATEGORY"), ("1tr", "AMOUNT")]),
        ("tăng mục tiêu tiết kiệm 2tr", [("tăng", "VERB"), ("mục tiêu", "TARGET"), ("tiết kiệm", "CATEGORY"), ("2tr", "AMOUNT")]),
        ("giảm giới hạn ăn uống xuống 500k", [("giảm", "VERB"), ("giới hạn", "TARGET"), ("ăn uống", "CATEGORY"), ("500k", "AMOUNT")]),
        ("order GrabFood bún bò 55k", [("GrabFood", "CATEGORY"), ("55k", "AMOUNT")]),
        ("lương tháng về 14tr", [("lương", "CATEGORY"), ("14tr", "AMOUNT")]),
        ("hoa hồng bán hàng 800k", [("hoa hồng", "CATEGORY"), ("800k", "AMOUNT")]),
    ]
    for t, plist in pairs:
        add(t, plist)

    for svc, money in [("Netflix", "109k"), ("Spotify", "59k"), ("CapCut", "79k")]:
        t = f"thanh toán {svc} tháng {money}"
        add(t, [(svc, "CATEGORY"), (money, "AMOUNT")])

    grab_ner = [
        ("đi grab hết 39k", [("grab", "CATEGORY"), ("39k", "AMOUNT")]),
        ("bắt grab 28k", [("grab", "CATEGORY"), ("28k", "AMOUNT")]),
        ("đi grab mất 28k", [("grab", "CATEGORY"), ("28k", "AMOUNT")]),
        ("mua hủ tiếu gõ 18k", [("hủ tiếu", "CATEGORY"), ("18k", "AMOUNT")]),
        ("hủ tiếu mua 38k", [("hủ tiếu", "CATEGORY"), ("38k", "AMOUNT")]),
        ("mua hủ tiếu 28k", [("hủ tiếu", "CATEGORY"), ("28k", "AMOUNT")]),
    ]
    for t, plist in grab_ner:
        add(t, plist)

    more_ner = [
        ("GrabFood ship cơm tấm 50k", [("GrabFood", "CATEGORY"), ("50k", "AMOUNT")]),
        ("GrabFood giao phở tái 44k", [("GrabFood", "CATEGORY"), ("44k", "AMOUNT")]),
        ("order GrabFood bún bò 62k", [("GrabFood", "CATEGORY"), ("62k", "AMOUNT")]),
        ("cuốc GrabBike đi làm 33k", [("GrabBike", "CATEGORY"), ("33k", "AMOUNT")]),
        ("GrabBike về nhà 27k", [("GrabBike", "CATEGORY"), ("27k", "AMOUNT")]),
        ("bật chế độ tối trong app", [("bật", "ACTION_TYPE"), ("chế độ tối", "TARGET")]),
        ("tắt dark mode giúp mình", [("tắt", "ACTION_TYPE"), ("dark mode", "TARGET")]),
        ("đặt hạn mức ăn uống 2tr", [("đặt", "VERB"), ("hạn mức", "TARGET"), ("ăn uống", "CATEGORY"), ("2tr", "AMOUNT")]),
        ("giao diện tối dễ nhìn khi làm đêm ở laptop", [("giao diện tối", "CATEGORY")]),
        ("ship đồ ăn qua GrabFood 88k", [("GrabFood", "CATEGORY"), ("88k", "AMOUNT")]),
        ("xe ôm grab 19k", [("grab", "CATEGORY"), ("19k", "AMOUNT")]),
        ("tiền grab bike sáng nay 25k", [("grab bike", "CATEGORY"), ("25k", "AMOUNT")]),
    ]
    for t, plist in more_ner:
        add(t, plist)

    return rows


def augment_record_transport_food_edges(existing: set[str]) -> list[dict]:
    """Grab / hủ tiếu: câu ngắn giống người dùng thật, nhãn category rõ (Transport / Food)."""
    rows: list[dict] = []
    amts = ["18", "22", "28", "35", "38", "39", "45", "52", "60"]
    transport_tpl = [
        "đi grab hết {a}k",
        "bắt grab {a}k",
        "đi grab {a}k",
        "đi grab mất {a}k",
        "book grab {a}k",
        "cuốc grab về nhà {a}k",
        "GrabBike {a}k",
        "đi Grab xe máy {a}k",
        "grab bike hết {a}k",
        "đặt grab đi làm {a}k",
        "đi grab ship {a}k",
        "chi grab hết {a}k",
        "thanh toán grab {a}k",
        "grab car {a}k",
        "di grab het {a}k",
    ]
    food_tpl = [
        "mua hủ tiếu gõ {a}k",
        "hủ tiếu mua {a}k",
        "mua hủ tiếu {a}k",
        "hủ tiếu Nam Vang {a}k",
        "tiệm hủ tiếu hết {a}k",
        "ăn hủ tiếu tái {a}k",
        "hủ tiếu khô {a}k",
        "hủ tiếu bò viên {a}k",
        "tô hủ tiếu {a}k",
        "ship hủ tiếu {a}k",
    ]
    for tpl in transport_tpl:
        for a in amts:
            text = tpl.format(a=a).strip()
            if text in existing:
                continue
            existing.add(text)
            rows.append({"text": text, "label": "Transport", "type": "expense", "is_money": 1})
    for tpl in food_tpl:
        for a in amts:
            text = tpl.format(a=a).strip()
            if text in existing:
                continue
            existing.add(text)
            rows.append({"text": text, "label": "Food", "type": "expense", "is_money": 1})
    return rows


def augment_action_dark_mode_edges(existing: set[str]) -> list[dict]:
    """
    Phân tách Action (UI) vs Chitchat: nhiều biến thể câu lệnh bật/tắt giao diện tối.
    Dùng action_type Setting (đồng dạng với \"Tắt chế độ tối\" đang dự đoán tốt).
    """
    rows: list[dict] = []
    phrases = [
        "bật chế độ tối",
        "Bật chế độ tối",
        "bật chế độ tối nhé",
        "bật chế độ tối giúp mình",
        "giúp tôi bật chế độ tối",
        "cho mình bật chế độ tối",
        "tôi muốn bật chế độ tối",
        "bật dark mode",
        "Bật dark mode",
        "bật night mode",
        "mở chế độ ban đêm",
        "kích hoạt dark mode",
        "kích hoạt giao diện tối",
        "bật giao diện tối",
        "bật giao diện tối đi",
        "switch sang dark mode",
        "bat che do toi",
        "bat dark mode",
        "mo che do toi",
        "tắt chế độ tối đi",
        "tắt dark mode giúp mình",
        "đóng chế độ tối",
        "tắt night mode",
        "bật lại chế độ tối sau khi cập nhật",
        "bật theme tối",
        "chuyển app sang chế độ tối",
    ]
    for text in phrases:
        t = text.strip()
        if t in existing:
            continue
        existing.add(t)
        rows.append({"text": t, "intent": "Action", "action_type": "Setting"})
    return rows


def augment_chitchat_evening_not_ui(existing: set[str]) -> list[dict]:
    """
    \"tối\" theo nghĩa thời gian / chào hỏi — không phải chế độ giao diện (tránh kéo nhầm sang UI).
    """
    rows: list[dict] = []
    lines = [
        ("chúc bạn buổi tối vui vẻ", "Neutral"),
        ("tối nay bạn rảnh không", "Neutral"),
        ("tối nay đi đâu chơi", "Neutral"),
        ("em chào anh buổi tối ạ", "Neutral"),
        ("chào buổi tối, hôm nay mệt quá", "Negative"),
        ("tối nay trời mát ghê", "Positive"),
        ("buổi tối tốt lành nhé", "Neutral"),
        ("tối qua ngủ không ngon", "Negative"),
        ("tối nay có đi làm không", "Neutral"),
        ("chúc ngủ ngon buổi tối", "Neutral"),
        ("hẹn tối mai gặp lại", "Neutral"),
        ("tối nay ăn gì hay", "Neutral"),
        ("mình chỉ muốn chào buổi tối thôi", "Neutral"),
        ("tối rồi đi ngủ sớm đi", "Neutral"),
        ("trời tối rồi nhỉ", "Neutral"),
    ]
    for text, sent in lines:
        t = text.strip()
        if t in existing:
            continue
        existing.add(t)
        rows.append({"text": t, "intent": "Chitchat", "sentiment": sent})
    return rows


def augment_chitchat_hard_negative_ui(existing: set[str]) -> list[dict]:
    """
    Hard negative: có từ dark mode / chế độ / theme / giao diện nhưng là chuyện trò chuyện,
    không phải lệnh bật tính năng app.
    """
    rows: list[dict] = []
    specs = [
        ("chế độ nói chuyện này của bot dễ thương quá ha", "Positive"),
        ("dark mode là gì vậy bạn giải thích chơi thôi", "Neutral"),
        ("mình thích theme sáng hơn theme tối khi đọc sách ở ngoài", "Neutral"),
        ("app kia bật dark mode xấu quá cười muốn xỉu", "Positive"),
        ("không cần bật chế độ gì đâu mình chỉ hỏi chơi thôi", "Neutral"),
        ("theme tối hay theme sáng dễ nhìn hơn theo bạn", "Neutral"),
        ("bạn nghĩ sao về giao diện tối của messenger zalo", "Neutral"),
        ("mình đang tám chuyện chứ không nhờ bạn bật tính năng đâu", "Neutral"),
        ("chế độ hài hước của bạn làm mình cười suốt", "Positive"),
        ("night mode trên điện thoại có tốn pin không nhỉ", "Neutral"),
        ("tối qua mình mơ thấy dark mode bay lơ lửng kỳ cục ghê", "Negative"),
        ("bạn có thích theme pastel không hay chỉ thích đen trắng", "Neutral"),
        ("giao diện tối của phim noir rất hợp vibe chứ không nói app đâu", "Neutral"),
        ("mình khen chế độ đùa của bạn thôi không phải lệnh đâu nha", "Positive"),
        ("dark mode meme trên mạng nhiều quá chán ghê", "Negative"),
        ("chỉ muốn chat về UX chung chung không cấu hình app", "Neutral"),
        ("bạn nghĩ chế độ sáng có hại mắt không khi làm đêm", "Neutral"),
        ("theme đẹp trên pinterest xem cho vui thôi", "Neutral"),
        ("mình đang phàn nàn giao diện web chứ không chỉnh app chi tiêu", "Negative"),
        ("kể chuyện dark mode hoá ra là trend tiktok thôi", "Neutral"),
        ("chế độ trả lời của bạn hơi máy móc nhưng cũng dễ thương", "Neutral"),
        ("hỏi cho vui: night mode có giúp ngủ ngon hơn không ta", "Neutral"),
        ("mình không muốn đổi cài đặt gì chỉ buồn muốn tâm sự", "Negative"),
        ("câu vừa rồi chỉ là ví dụ ngữ pháp có chữ chế độ thôi", "Neutral"),
        ("bạn hiểu nhầm rồi mình không bảo bật dark mode đâu", "Neutral"),
    ]
    for text, sent in specs:
        t = text.strip()
        if t in existing:
            continue
        existing.add(t)
        rows.append({"text": t, "intent": "Chitchat", "sentiment": sent})
    return rows


def augment_chitchat_emotional_clear(existing: set[str]) -> list[dict]:
    """Câu cảm xúc rõ (Positive / Negative / Neutral) để encoder sentiment bớt lệch Neutral."""
    rows: list[dict] = []
    specs = [
        ("Mình vui quá đi mất thôi", "Positive"),
        ("Cảm ơn bạn nhiều lắm luôn", "Positive"),
        ("Hôm nay trúng mini game vui ghê", "Positive"),
        ("Thích app của bạn quá đi mất", "Positive"),
        ("Bạn trả lời hay quá mình muốn share cho bạn bè", "Positive"),
        ("Đọc xong mình thấy nhẹ lòng hẳn", "Positive"),
        ("Chúc team làm app ngày càng thành công nha", "Positive"),
        ("Hôm nay mệt bở hơi tai luôn", "Negative"),
        ("Stress quá không biết làm sao", "Negative"),
        ("Thất vọng ghê chuyện gì cũng hỏng", "Negative"),
        ("Buồn ngủ mà deadline dí sát cổ", "Negative"),
        ("Mình hơi chán không muốn làm gì cả", "Negative"),
        ("Cảm giác cô đơn quá đi mất", "Negative"),
        ("Sợ không đủ tiền cuối tháng quá", "Negative"),
        ("Hôm nay là thứ mấy vậy bạn", "Neutral"),
        ("Cho mình hỏi tí thôi nhé", "Neutral"),
        ("Bạn có khỏe không", "Neutral"),
        ("Ừm mình đang test chat", "Neutral"),
        ("Ok cảm ơn để mình xem lại", "Neutral"),
        ("Biết rồi để đó đi", "Neutral"),
        ("Mình chỉ đang lướt app cho đỡ buồn", "Neutral"),
        ("Không có gì quan trọng đâu", "Neutral"),
        ("Bạn là bot hay người thật vậy", "Neutral"),
        ("Giờ mấy giờ rồi nhỉ", "Neutral"),
        ("Hôm nay nắng hay mưa ta", "Neutral"),
        ("Mình mới uống cà phê xong tinh thần phơi phới", "Positive"),
        ("Được nghỉ lễ mình sướng quá trời luôn", "Positive"),
        ("Bị trừ lương oan mình tức muốn khóc", "Negative"),
        ("Cãi nhau với sếp mệt nghỉ luôn", "Negative"),
        ("Nghe nhạc chill một mình cũng được", "Neutral"),
        ("Đi dạo một mình cho khuây khỏa", "Neutral"),
        ("Năm mới chúc mọi người an khang thịnh vượng", "Positive"),
        ("Tết xa nhà hơi buồn xíu", "Negative"),
        ("Ping thử xem bot có online không", "Neutral"),
        ("Random chat thôi đừng ghi chi tiêu giùm", "Neutral"),
    ]
    for text, sent in specs:
        t = text.strip()
        if t in existing:
            continue
        existing.add(t)
        rows.append({"text": t, "intent": "Chitchat", "sentiment": sent})
    return rows


def augment_record_grabfood_delivery(existing: set[str]) -> list[dict]:
    """GrabFood / ship đồ ăn → Food; tách khỏi cuốc xe (Transport)."""
    rows: list[dict] = []
    dishes = [
        "cơm tấm sườn",
        "bún bò Huế",
        "phở tái",
        "bánh mì thịt",
        "gà rán",
        "trà sữa full topping",
        "mì cay",
        "bún riêu",
    ]
    amts = ["28", "35", "42", "48", "55", "62", "88", "99"]
    tpls = [
        "GrabFood {dish} {a}k",
        "order GrabFood {dish} {a}k",
        "ship GrabFood {dish} hết {a}k",
        "đặt món GrabFood {dish} {a}k",
        "GrabFood ship {dish} {a}k",
        "mua đồ ăn GrabFood {dish} {a}k",
        "GrabFood giao {dish} {a}k",
        "app GrabFood {dish} tính {a}k",
    ]
    for dish in dishes:
        for a in amts:
            for tpl in tpls:
                text = tpl.format(dish=dish, a=a).strip()
                if text in existing:
                    continue
                existing.add(text)
                rows.append({"text": text, "label": "Food", "type": "expense", "is_money": 1})
    return rows


def augment_record_cafe_and_record_vs_action_edges(existing: set[str]) -> list[dict]:
    rows: list[dict] = []
    # Cafe - Food vs Entertainment
    food_cafe = [
        "mua cà phê sữa đá {a}",
        "mua cafe sữa đá {a}",
        "mua cafe mang đi {a}",
        "mua hạt cà phê {a}",
        "order 2 ly cafe {a}",
        "order cà phê mang về {a}",
        "mua 1 ly cafe sữa {a}",
        "mua cafe đen {a}",
    ]
    ent_cafe = [
        "đi cà phê với bạn bè hết {a}",
        "đi cafe với bạn {a}",
        "đi cf với người yêu {a}",
        "tối đi uống cà phê với bạn {a}",
        "đi cafe sữa đá với nhóm bạn {a}",
        "đi uống cà phê tám chuyện {a}",
        "hẹn đi cafe sữa đá {a}",
        "tụ tập đi cf {a}",
    ]
    amts = ["19k", "25k", "35k", "40k", "50k", "80k", "90k", "120k"]
    for tpl in food_cafe:
        for a in amts:
            text = tpl.format(a=a).strip()
            if text not in existing:
                existing.add(text)
                rows.append({"text": text, "label": "Food", "type": "expense", "is_money": 1})
    for tpl in ent_cafe:
        for a in amts:
            text = tpl.format(a=a).strip()
            if text not in existing:
                existing.add(text)
                rows.append({"text": text, "label": "Entertainment", "type": "expense", "is_money": 1})

    # Record vs Action hard negatives
    record_negatives = [
        "mới tiêu {a}",
        "đã tiêu {a} rồi",
        "hôm nay tiêu hết {a}",
        "ghi chép {a} chi tiêu",
        "tiêu {a} ăn uống",
        "chi tiêu hết {a}",
        "mua đồ hết {a}",
        "tiêu hết {a} cho ăn uống",
        "thanh toán {a} tiền nhà",
        "đóng {a} tiền học",
    ]
    amts_large = ["200k", "500k", "1tr", "2tr", "3tr", "5tr", "10tr", "1.5 triệu", "2 triệu", "3 triệu", "5 triệu", "10 triệu"]
    for tpl in record_negatives:
        for a in amts_large:
            text = tpl.format(a=a).strip()
            if text not in existing:
                existing.add(text)
                lab = "Food"
                if "tiền nhà" in text:
                    lab = "Housing"
                elif "tiền học" in text:
                    lab = "Education"
                rows.append({"text": text, "label": lab, "type": "expense", "is_money": 1})
    return rows


def augment_action_limits_and_operators(existing: set[str]) -> list[dict]:
    rows: list[dict] = []
    limit_actions = [
        "Đặt giới hạn chi tiêu thành {a}",
        "cài đặt hạn mức chi tiêu {a}",
        "đặt giới hạn ăn uống {a}",
        "thay đổi hạn mức chi tiêu thành {a}",
        "cài hạn mức ăn uống là {a}",
        "đặt hạn mức chi tiêu thành {a}",
        "đặt giới hạn chi tiêu là {a}",
        "thiết lập hạn mức ăn uống thành {a}",
        "thiết lập giới hạn chi tiêu {a}",
        "chốt hạn mức chi tiêu {a}",
        "đặt lại giới hạn thành {a}",
        "Thêm {a} vào ăn uống",
        "cộng thêm {a} vào hạn mức đi lại",
        "tăng thêm {a} cho mua sắm",
        "bù vào mục tiêu tiết kiệm {a}",
        "bổ sung {a} vào hạn mức",
        "tăng hạn mức ăn uống lên {a}",
        "bớt {a} từ giới hạn giải trí",
        "giảm {a} hạn mức đi lại",
        "trừ đi {a} hạn mức ăn uống",
        "giảm giới hạn ăn uống xuống {a}",
    ]
    amts_large = ["200k", "500k", "1tr", "2tr", "3tr", "5tr", "10tr", "1.5 triệu", "2 triệu", "3 triệu", "5 triệu", "10 triệu"]
    for tpl in limit_actions:
        for a in amts_large:
            text = tpl.format(a=a).strip()
            if text not in existing:
                existing.add(text)
                rows.append({"text": text, "intent": "Action", "action_type": "SET_LIMIT"})
    return rows


def main() -> None:
    print("=== Fix disambiguation labels (mua/bán, cafe, gạo…) ===")
    from fix_disambiguation_labels import main as fix_disambiguation_main
    from boost_action_intent_rows import main as boost_action_main

    fix_disambiguation_main()
    boost_action_main()

    print("=== Dedupe + augment datasets ===")
    rec = pd.read_csv(RECORD_CSV, encoding="utf-8-sig")
    print("intent_record before", len(rec))
    rec = dedupe_record(rec)
    ex_r = set(rec["text"].astype(str).str.strip())
    add_r = augment_record(ex_r)
    ex_r |= {r["text"] for r in add_r}
    add_r.extend(augment_record_transport_food_edges(ex_r))
    ex_r |= {r["text"] for r in add_r}
    add_r.extend(augment_record_grabfood_delivery(ex_r))
    ex_r |= {r["text"] for r in add_r}
    add_r.extend(augment_record_cafe_and_record_vs_action_edges(ex_r))
    if add_r:
        rec = pd.concat([rec, pd.DataFrame(add_r)], ignore_index=True)
    rec.to_csv(RECORD_CSV, index=False, encoding="utf-8-sig")
    print("intent_record after", len(rec))

    act = pd.read_csv(ACTION_CSV, encoding="utf-8-sig")
    act = normalize_action_types_column(act)
    print("intent_action before", len(act))
    act = dedupe_action(act)
    ex_a = set(act["text"].astype(str).str.strip())
    add_a = augment_action(ex_a)
    ex_a |= {r["text"] for r in add_a}
    add_a.extend(augment_action_dark_mode_edges(ex_a))
    ex_a |= {r["text"] for r in add_a}
    add_a.extend(augment_action_limits_and_operators(ex_a))
    if add_a:
        act = pd.concat([act, pd.DataFrame(add_a)], ignore_index=True)
    act.to_csv(ACTION_CSV, index=False, encoding="utf-8-sig")
    print("intent_action after", len(act))

    chat = pd.read_csv(CHITCHAT_CSV, encoding="utf-8-sig")
    print("intent_chitchat before", len(chat))
    chat = dedupe_chitchat(chat)
    ex_c = set(chat["text"].astype(str).str.strip())
    add_c = augment_chitchat(ex_c)
    ex_c |= {r["text"] for r in add_c}
    add_c.extend(augment_chitchat_evening_not_ui(ex_c))
    ex_c |= {r["text"] for r in add_c}
    add_c.extend(augment_chitchat_hard_negative_ui(ex_c))
    ex_c |= {r["text"] for r in add_c}
    add_c.extend(augment_chitchat_emotional_clear(ex_c))
    if add_c:
        chat = pd.concat([chat, pd.DataFrame(add_c)], ignore_index=True)
    chat.to_csv(CHITCHAT_CSV, index=False, encoding="utf-8-sig")
    print("intent_chitchat after", len(chat))

    ner_lines: list[dict] = []
    with NER_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ner_lines.append(json.loads(line))
    print("ner_dataset before", len(ner_lines))
    ner_lines = dedupe_ner_lines(ner_lines)
    ex_n = {row["text"].strip() for row in ner_lines}
    add_n = augment_ner(ex_n)
    ner_lines.extend(add_n)
    with NER_JSONL.open("w", encoding="utf-8") as f:
        for row in ner_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("ner_dataset after", len(ner_lines), "(added", len(add_n), ")")

    print("Done.")


if __name__ == "__main__":
    main()
